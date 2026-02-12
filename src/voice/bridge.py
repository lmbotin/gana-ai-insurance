"""
Audio Bridge: Coordinates audio flow between Twilio and OpenAI Realtime API.

This module handles:
- Bidirectional audio streaming (Twilio <-> OpenAI)
- Barge-in detection and handling
- Operational liability claim state management during the call
- Transcript extraction and field population
- Claim completeness checking and guided information gathering
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket  # type: ignore[import-untyped]

from ..fnol.extractor import OperationalClaimExtractor
from ..fnol.state_manager import OperationalClaimStateManager
from ..fnol.checker import check_claim, CheckReport
from ..policy import get_policy_service
from .openai_realtime import OpenAIRealtimeClient
from .prompts import get_voice_agent_prompt, CLAIM_COMPLETE_PROMPT

logger = logging.getLogger(__name__)


# Minimum completeness threshold to consider claim sufficient
# Higher threshold ensures more complete information before ending
COMPLETENESS_THRESHOLD = 0.75


class AudioBridge:
    """
    Bridges audio between Twilio Media Streams and OpenAI Realtime API.

    Manages the full lifecycle of an operational liability claim intake call, including:
    - Audio forwarding in both directions
    - Barge-in (interruption) handling
    - Claim state tracking and updates
    - Transcript extraction for field population
    """
    
    # Type alias for callback
    OnCallStartedCallback = Optional[callable]
    
    def __init__(self, call_sid: Optional[str] = None, on_call_started: OnCallStartedCallback = None):
        """
        Initialize the audio bridge.
        
        Args:
            call_sid: Twilio Call SID for this call
            on_call_started: Optional callback called when call_sid is known.
                             Signature: callback(call_sid: str, bridge: AudioBridge)
        """
        self.call_sid = call_sid
        self.stream_sid: Optional[str] = None
        self._on_call_started = on_call_started
        
        # Initialize components
        self.claim_state = OperationalClaimStateManager(call_sid=call_sid)
        self.openai_client = OpenAIRealtimeClient()
        self.extractor = OperationalClaimExtractor()
        
        # Keep backwards compatibility alias
        self.fnol_state = self.claim_state
        
        # State tracking
        self._twilio_ws: Optional[WebSocket] = None
        self._is_agent_speaking = False
        self._pending_transcripts: list[str] = []
        self._extraction_task: Optional[asyncio.Task] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        
        # Completeness tracking
        self._last_check_report: Optional[CheckReport] = None
        self._claim_complete_notified = False
        
        # Call ending - track goodbye from both parties
        self._should_end_call = False
        self._agent_said_goodbye = False
        self._user_said_goodbye = False
        self._goodbye_timeout_task: Optional[asyncio.Task] = None
        
        # Register OpenAI event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for the OpenAI client."""
        self.openai_client.on_speech_started(self._handle_speech_started)
        self.openai_client.on_transcript(self._handle_transcript)
        self.openai_client.on_error(self._handle_error)
    
    async def run(self, twilio_ws: WebSocket) -> None:
        """
        Run the audio bridge for a call.
        
        Args:
            twilio_ws: The Twilio WebSocket connection
        """
        self._twilio_ws = twilio_ws
        self._shutdown_event = asyncio.Event()
        
        try:
            async with self.openai_client.connect():
                logger.info(f"Audio bridge started for call {self.call_sid}")
                
                # Send initial greeting to start the conversation
                await self._send_initial_greeting()
                
                # Run both directions concurrently
                tasks = [
                    asyncio.create_task(self._twilio_to_openai()),
                    asyncio.create_task(self._openai_to_twilio()),
                ]
                
                # Wait for either task to complete (usually Twilio "stop" event)
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks when one completes
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check for exceptions in completed tasks
                for task in done:
                    if task.exception():
                        logger.error(f"Task error: {task.exception()}")
                
        except Exception as e:
            logger.error(f"Audio bridge error: {e}")
            raise
        finally:
            # Finalize the FNOL state
            final_state = self.fnol_state.finalize()
            logger.info(
                f"Call ended. FNOL completion: {self.fnol_state.get_completion_percentage():.0f}%"
            )
            logger.debug(f"Final FNOL state: {json.dumps(final_state, indent=2)}")
    
    async def _send_initial_greeting(self) -> None:
        """Send initial greeting to start the conversation."""
        try:
            # Give the session a moment to fully initialize
            await asyncio.sleep(0.5)
            
            # Create the initial response to greet the caller
            await self.openai_client.create_response()
            logger.info("Initial greeting requested from OpenAI")
        except Exception as e:
            logger.error(f"Failed to send initial greeting: {e}")
    
    async def _twilio_to_openai(self) -> None:
        """Forward audio from Twilio to OpenAI."""
        try:
            while True:
                try:
                    message = await self._twilio_ws.receive_text()
                except Exception as e:
                    logger.info(f"Twilio WebSocket closed: {e}")
                    break
                    
                data = json.loads(message)
                event = data.get("event")
                
                if event == "start":
                    # Call started - extract metadata
                    start_data = data.get("start", {})
                    self.stream_sid = start_data.get("streamSid")
                    self.call_sid = start_data.get("callSid", self.call_sid)
                    
                    # Update claim state with call info
                    self.claim_state.call_sid = self.call_sid
                    self.claim_state.stream_sid = self.stream_sid
                    
                    logger.info(f"Twilio stream started: {self.stream_sid}")
                    
                    # Notify that call has started (for registration in active_calls)
                    if self._on_call_started and self.call_sid:
                        self._on_call_started(self.call_sid, self)
                    
                elif event == "media":
                    # Audio data from caller
                    media = data.get("media", {})
                    payload = media.get("payload")
                    
                    if payload:
                        # Forward audio to OpenAI
                        # Twilio sends base64-encoded G.711 μ-law audio
                        # OpenAI Realtime is configured to accept g711_ulaw
                        await self.openai_client.send_audio(payload)
                        
                elif event == "stop":
                    # Call ended
                    logger.info("Twilio stream stopped")
                    break
                    
                elif event == "mark":
                    # Playback mark reached
                    logger.debug(f"Playback mark: {data.get('mark', {}).get('name')}")
                    
        except asyncio.CancelledError:
            logger.info("Twilio->OpenAI stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Twilio->OpenAI stream: {e}")
    
    async def _openai_to_twilio(self) -> None:
        """Forward audio from OpenAI to Twilio."""
        try:
            async for event in self.openai_client.receive_events():
                # Handle session events (for debugging)
                if event.type == "session.created":
                    logger.info("OpenAI Realtime session created")
                elif event.type == "session.updated":
                    logger.info("OpenAI Realtime session configured")
                
                # Handle audio output
                elif event.is_audio_delta and event.audio_delta:
                    self._is_agent_speaking = True
                    await self._send_audio_to_twilio(event.audio_delta)
                
                # Handle response completion
                elif event.type == "response.audio.done":
                    self._is_agent_speaking = False
                    # Process any pending transcripts now that agent is done speaking
                    await self._process_pending_transcripts()
                
                # Handle response done
                elif event.type == "response.done":
                    logger.debug("OpenAI response completed")
                    # Check if both parties have said goodbye
                    if self._should_end_call and self._agent_said_goodbye and self._user_said_goodbye:
                        logger.info("Both parties said goodbye - ending call now")
                        await asyncio.sleep(1.0)  # Brief pause after mutual goodbye
                        await self._end_twilio_call()
                        break
                    elif self._should_end_call and self._agent_said_goodbye:
                        # Agent said goodbye, wait a bit for user's response
                        logger.info("Agent said goodbye, waiting for user response...")
                        # Don't break - wait for user's goodbye
                
                # Handle AI's response transcript (to detect goodbye and END_CALL)
                elif event.type == "response.audio_transcript.done":
                    transcript = event.data.get("transcript", "")
                    if transcript:
                        # Add to conversation record
                        self.fnol_state.add_transcript_entry("assistant", transcript)
                        logger.debug(f"Agent said: {transcript[:100]}...")
                        
                        transcript_lower = transcript.lower()
                        
                        # Check for END_CALL signal (case insensitive, anywhere in text)
                        if "END_CALL" in transcript.upper() or "END CALL" in transcript.upper():
                            logger.info("Agent sent END_CALL signal")
                            self._should_end_call = True
                        
                        # Check if agent said goodbye
                        goodbye_phrases = ["bye", "goodbye", "take care", "have a good", "talk soon"]
                        if any(phrase in transcript_lower for phrase in goodbye_phrases):
                            logger.info("Agent said goodbye")
                            self._agent_said_goodbye = True
                            # If user already said goodbye, we can end
                            if self._user_said_goodbye:
                                self._should_end_call = True
                            else:
                                # Start a timeout - if user doesn't respond in 5 seconds, end anyway
                                if self._goodbye_timeout_task is None:
                                    self._goodbye_timeout_task = asyncio.create_task(
                                        self._goodbye_timeout()
                                    )
                
                # Handle speech started (barge-in)
                elif event.is_speech_started:
                    if self._is_agent_speaking:
                        logger.info("🛑 User interrupted - stopping agent speech")
                        # User interrupted - clear Twilio playback immediately
                        await self._clear_twilio_playback()
                        # Cancel OpenAI response
                        await self.openai_client.cancel_response()
                        self._is_agent_speaking = False
                        # Note: The AI will naturally respond to the interruption
                        # because the prompt instructs it to say "Oh, go ahead" etc.
                
                # Handle transcript completion
                elif event.is_transcript_complete and event.transcript:
                    # Queue transcript for processing
                    self._pending_transcripts.append(event.transcript)
                    # Add to conversation record
                    self.fnol_state.add_transcript_entry("user", event.transcript)
                    
                    # Check if user said goodbye
                    user_text_lower = event.transcript.lower()
                    goodbye_phrases = ["bye", "goodbye", "take care", "thank you", "thanks"]
                    if any(phrase in user_text_lower for phrase in goodbye_phrases):
                        logger.info(f"User said goodbye: '{event.transcript}'")
                        self._user_said_goodbye = True
                        # If agent already said goodbye, we can end
                        if self._agent_said_goodbye:
                            self._should_end_call = True
                            logger.info("Both parties have said goodbye - will end call after next response")
                    
                    # If agent isn't speaking, process immediately
                    if not self._is_agent_speaking:
                        await self._process_pending_transcripts()
                
                # Handle errors
                elif event.is_error:
                    logger.error(f"OpenAI error: {event.error_message}")
                    
        except asyncio.CancelledError:
            logger.info("OpenAI->Twilio stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI->Twilio stream: {e}")
    
    async def _send_audio_to_twilio(self, audio_b64: str) -> None:
        """Send audio data to Twilio."""
        if not self._twilio_ws or not self.stream_sid:
            return
        
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": audio_b64,
            },
        }
        
        try:
            await self._twilio_ws.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send audio to Twilio: {e}")
    
    async def _clear_twilio_playback(self) -> None:
        """Clear Twilio's playback buffer (for barge-in)."""
        if not self._twilio_ws or not self.stream_sid:
            return
        
        message = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        
        try:
            await self._twilio_ws.send_text(json.dumps(message))
            logger.debug("Cleared Twilio playback buffer")
        except Exception as e:
            logger.error(f"Failed to clear Twilio playback: {e}")
    
    async def _end_twilio_call(self) -> None:
        """End the Twilio call gracefully."""
        # Cancel any pending goodbye timeout
        if self._goodbye_timeout_task and not self._goodbye_timeout_task.done():
            self._goodbye_timeout_task.cancel()
        
        if not self._twilio_ws:
            return
        
        try:
            # Close the WebSocket which will end the Media Stream
            await self._twilio_ws.close()
            logger.info("Twilio call ended gracefully")
        except Exception as e:
            logger.warning(f"Error closing Twilio WebSocket: {e}")
    
    async def _goodbye_timeout(self) -> None:
        """Timeout handler - end call if user doesn't respond after agent says goodbye."""
        try:
            await asyncio.sleep(5.0)  # Wait 5 seconds for user to respond
            if self._agent_said_goodbye and not self._user_said_goodbye:
                logger.info("Goodbye timeout - user didn't respond, ending call")
                self._should_end_call = True
                self._user_said_goodbye = True  # Force the condition to end
                await self._end_twilio_call()
        except asyncio.CancelledError:
            pass  # Timeout was cancelled (user responded)
    
    async def _process_pending_transcripts(self) -> None:
        """Process pending transcripts for claim field extraction."""
        if not self._pending_transcripts:
            return
        
        # Combine pending transcripts
        combined = " ".join(self._pending_transcripts)
        self._pending_transcripts.clear()
        
        logger.info(f"📝 Processing user transcript: {combined[:100]}...")
        
        # Extract fields from transcript
        try:
            current_state = self.claim_state.to_dict()
            context = self.claim_state._transcript[-4:]  # Last 4 turns
            
            extracted = await self.extractor.extract(
                transcript=combined,
                current_state=current_state,
                conversation_context=context,
            )
            
            if extracted:
                logger.info(f"🔍 Extracted fields: {extracted}")
                updated_fields = self.claim_state.apply_patch(extracted)
                logger.info(f"✅ Updated claim fields: {updated_fields}")
                
                # Log current state for debugging
                claim = self.claim_state.claim
                logger.debug(f"Current claim state - Name: {claim.claimant.name}, Policy: {claim.claimant.policy_number}")
                
                # Update the agent's instructions with new context
                await self._update_agent_context()
            else:
                logger.debug("No fields extracted from this transcript")
                
        except Exception as e:
            logger.error(f"❌ Failed to extract claim fields: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_claim_completeness(self) -> CheckReport:
        """
        Check the current claim for completeness using the checker module.
        
        Returns:
            CheckReport with completeness score, missing evidence, and recommendations.
        """
        claim = self.claim_state.claim
        report = check_claim(claim)
        self._last_check_report = report
        
        logger.info(
            f"Claim completeness: {report.completeness_score:.0%}, "
            f"missing: {len(report.missing_required_evidence)} items, "
            f"contradictions: {len(report.contradictions)}"
        )
        
        return report
    
    def is_claim_sufficient(self) -> bool:
        """
        Check if the claim has enough information to proceed.
        
        Returns:
            True if completeness score >= COMPLETENESS_THRESHOLD
        """
        report = self._check_claim_completeness()
        return report.completeness_score >= COMPLETENESS_THRESHOLD
    
    async def _update_agent_context(self) -> None:
        """
        Update the agent's system prompt with current claim state.
        
        Includes completeness checking to guide the conversation toward
        collecting missing critical information.
        """
        # Run completeness check
        report = self._check_claim_completeness()
        
        # Get state manager's view of missing fields
        missing = self.claim_state.get_missing_fields()
        missing_ids = [f["id"] for f in missing[:5]]  # First 5 missing

        # Policy check: if we have name + policy number, verify against policy DB
        policy_issue = None
        claim = self.claim_state.claim
        if getattr(claim.claimant, "policy_number", None) and getattr(claim.claimant, "name", None):
            try:
                svc = get_policy_service()
                policy = svc.get_policy(str(claim.claimant.policy_number))
                if not policy:
                    policy_issue = "Policy not found. Ask the caller to confirm their policy number."
                elif not svc.verify_claimant_name(policy, claim.claimant.name):
                    policy_issue = f"The name on the policy is '{policy.named_insured}'. Please confirm with the caller that they are the policyholder."
            except Exception as e:
                logger.debug(f"Policy check skipped: {e}")

        # Also include checker's recommended questions
        recommended_questions = report.recommended_questions

        next_q = self.claim_state.get_next_question()
        next_question = next_q["question"] if next_q else None
        # When there is a policy issue, prioritize resolving it before other questions
        if policy_issue:
            next_question = (
                "Address the policy issue with the caller first (confirm policy number or name). "
                "Do not ask about damage or incident details until this is resolved."
            )

        # If claim is sufficiently complete, switch to wrap-up mode
        if report.completeness_score >= COMPLETENESS_THRESHOLD and not self._claim_complete_notified:
            self._claim_complete_notified = True
            logger.info(
                f"Claim reached sufficient completeness ({report.completeness_score:.0%}). "
                "Switching to wrap-up mode."
            )
            
            # Use the claim complete prompt
            new_prompt = CLAIM_COMPLETE_PROMPT
            if policy_issue:
                new_prompt += f"\n\nPOLICY CHECK: {policy_issue}"

            # Add any remaining recommendations
            if recommended_questions:
                new_prompt += f"\n\nOPTIONAL FOLLOW-UPS (only if time permits):\n"
                for q in recommended_questions[:2]:
                    new_prompt += f"- {q}\n"
        else:
            # Build prompt with completeness context and any policy check message
            new_prompt = get_voice_agent_prompt(
                missing_fields=missing_ids,
                next_question=next_question,
                policy_issue=policy_issue,
            )
            
            # Add completeness status
            new_prompt += f"\n\nCLAIM STATUS: {report.completeness_score:.0%} complete"
            
            # Add critical missing items from checker
            if report.missing_required_evidence:
                critical_missing = [
                    item for item in report.missing_required_evidence
                    if item in ("damage_photos", "incident_description", "damage_type", "property_type")
                ]
                if critical_missing:
                    new_prompt += f"\nCRITICAL MISSING: {', '.join(critical_missing[:3])}"
            
            # Add recommended questions from checker
            if recommended_questions and recommended_questions[0] != next_question:
                new_prompt += f"\nALTERNATIVE QUESTION: {recommended_questions[0]}"
            
            # Add contradiction warnings
            if report.contradictions:
                new_prompt += f"\n\nWARNING - CONTRADICTIONS DETECTED:"
                for contradiction in report.contradictions[:2]:
                    new_prompt += f"\n- {contradiction}"
                new_prompt += "\nPlease gently clarify these discrepancies with the caller."
        
        try:
            await self.openai_client.update_instructions(new_prompt)
            logger.debug(f"Updated agent instructions (completeness: {report.completeness_score:.0%})")
        except Exception as e:
            logger.warning(f"Failed to update agent instructions: {e}")
    
    def _handle_speech_started(self) -> None:
        """Handle user speech start event."""
        logger.debug("User started speaking")
    
    def _handle_transcript(self, transcript: str) -> None:
        """Handle completed transcript."""
        logger.info(f"User said: {transcript[:100]}...")
    
    def _handle_error(self, error: str) -> None:
        """Handle OpenAI error."""
        logger.error(f"OpenAI Realtime error: {error}")
    
    def get_fnol_summary(self) -> str:
        """Get a summary of the collected FNOL information."""
        return self.fnol_state.get_summary()
    
    def get_fnol_data(self) -> dict:
        """Get the current FNOL data as a dictionary."""
        data = self.fnol_state.to_dict()
        
        # Add completeness information
        if self._last_check_report:
            data["_completeness"] = {
                "score": self._last_check_report.completeness_score,
                "missing_evidence": self._last_check_report.missing_required_evidence,
                "contradictions": self._last_check_report.contradictions,
                "recommended_questions": self._last_check_report.recommended_questions,
            }
        
        return data
    
    def get_check_report(self) -> Optional[CheckReport]:
        """
        Get the last completeness check report.
        
        Returns:
            CheckReport if a check has been performed, None otherwise.
        """
        return self._last_check_report
    
    def get_completeness_score(self) -> float:
        """
        Get the current claim completeness score.
        
        Returns:
            Completeness score between 0.0 and 1.0
        """
        report = self._check_claim_completeness()
        return report.completeness_score
