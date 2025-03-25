import base64
import requests
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
import time

# Initialize OpenAI client
client = OpenAI()

# Define emotion states and response strategies
EMOTION_STATES = ["neutral", "happy", "sad", "angry", "anxious", "surprised"]

# Response strategy library based on empirical psychology
RESPONSE_STRATEGIES = {
    "neutral": {
        "name": "information_focused",
        "description": "Provide clear, factual information with a balanced tone."
    },
    "happy": {
        "name": "positive_reinforcement",
        "description": "Match the positive energy, affirm feelings, and build on the positive state."
    },
    "sad": {
        "name": "empathetic_support",
        "description": "Express understanding, validate feelings, offer gentle encouragement and support."
    },
    "angry": {
        "name": "de_escalation",
        "description": "Acknowledge frustration, use calming language, avoid defensiveness, and focus on solutions."
    },
    "anxious": {
        "name": "reassurance",
        "description": "Provide calm reassurance, clear information to reduce uncertainty, and gentle guidance."
    },
    "surprised": {
        "name": "clarification",
        "description": "Provide context, explain unexpected information, and help process the surprising elements."
    }
}

# Emotion transition strategy matrix
TRANSITION_STRATEGIES = {
    ("neutral", "angry"): {
        "name": "preemptive_de_escalation",
        "description": "Recognize rising tension, acknowledge concerns early, and address issues before frustration increases."
    },
    ("neutral", "sad"): {
        "name": "supportive_response",
        "description": "Offer empathy and emotional support for the shift to sadness."
    },
    ("neutral", "anxious"): {
        "name": "structured_reassurance",
        "description": "Provide clear information and structured guidance to address emerging anxiety."
    },
    ("happy", "neutral"): {
        "name": "engagement_maintenance",
        "description": "Maintain engagement while adapting to more neutral emotional tone."
    },
    ("happy", "sad"): {
        "name": "gentle_transition",
        "description": "Acknowledge the mood shift with empathy, validate the change in emotions."
    },
    ("sad", "neutral"): {
        "name": "positive_refocusing",
        "description": "Gently guide toward constructive topics while respecting previous emotional state."
    },
    ("angry", "neutral"): {
        "name": "solution_focused",
        "description": "Capitalize on calming by moving toward constructive problem-solving."
    },
    ("anxious", "neutral"): {
        "name": "confidence_building",
        "description": "Reinforce security and emphasize capability as anxiety diminishes."
    },
    # Additional common transitions
    ("happy", "angry"): {
        "name": "disappointment_management",
        "description": "Address the shift from positive to negative by acknowledging disappointment and offering constructive paths forward."
    },
    ("sad", "happy"): {
        "name": "positive_reinforcement",
        "description": "Reinforce and encourage the positive emotional shift while acknowledging the previous state."
    },
    ("angry", "sad"): {
        "name": "empathetic_redirection",
        "description": "Acknowledge the underlying hurt beneath anger, offer compassion and supportive understanding."
    },
    ("anxious", "angry"): {
        "name": "frustration_validation",
        "description": "Validate how anxiety can transform to frustration, while calmly addressing concerns."
    }
}

class EmotionTracker:
    def __init__(self):
        self.emotion_history = []
        self.transition_counts = {}
        self.transition_timestamps = []
        
    def add_emotion(self, emotion: str):
        """Add a new emotion to the history and track transitions"""
        self.emotion_history.append(emotion)
        timestamp = time.time()
        
        # Track transitions
        if len(self.emotion_history) > 1:
            prev = self.emotion_history[-2]
            curr = self.emotion_history[-1]
            
            # FIX 2: Use string keys instead of tuples for transitions
            transition = f"{prev}_to_{curr}"  # Change from tuple to string format
            
            if transition in self.transition_counts:
                self.transition_counts[transition] += 1
            else:
                self.transition_counts[transition] = 1
                
            # Record timestamp of transition
            self.transition_timestamps.append((transition, timestamp))
    
    def get_most_frequent_transition(self):
        """Get the most frequently occurring emotion transition"""
        if not self.transition_counts:
            return ("none_to_none", 0)
        
        most_freq = max(self.transition_counts.items(), key=lambda x: x[1])
        return most_freq
    
    def get_dominant_emotion(self) -> str:
        """Get the most frequently expressed emotion"""
        if not self.emotion_history:
            return "none"
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def get_recent_trend(self, window_size=3) -> str:
        """Analyze recent emotion trend in the conversation"""
        if len(self.emotion_history) < window_size:
            return "insufficient_data"
            
        recent_emotions = self.emotion_history[-window_size:]
        
        # Check if emotions are consistent
        if all(emotion == recent_emotions[0] for emotion in recent_emotions):
            return f"consistent_{recent_emotions[0]}"
            
        # Check for improving trend (e.g., angry -> neutral -> happy)
        emotion_valence = {
            "angry": 1, 
            "sad": 2, 
            "anxious": 3, 
            "neutral": 4, 
            "surprised": 5, 
            "happy": 6
        }
        
        if all(emotion_valence.get(recent_emotions[i], 0) < emotion_valence.get(recent_emotions[i+1], 0) 
              for i in range(len(recent_emotions)-1)):
            return "improving"
            
        if all(emotion_valence.get(recent_emotions[i], 0) > emotion_valence.get(recent_emotions[i+1], 0) 
              for i in range(len(recent_emotions)-1)):
            return "deteriorating"
            
        return "fluctuating"
    
    def generate_transition_report(self) -> Dict:
        """Generate a comprehensive report of emotion transitions"""
        # Calculate time spent in each emotion
        emotion_durations = {}
        prev_timestamp = None
        prev_emotion = None
        
        for i, emotion in enumerate(self.emotion_history):
            timestamp = self.transition_timestamps[i-1][1] if i > 0 else time.time()
            
            if prev_timestamp and prev_emotion:
                duration = timestamp - prev_timestamp
                if prev_emotion in emotion_durations:
                    emotion_durations[prev_emotion] += duration
                else:
                    emotion_durations[prev_emotion] = duration
                    
            prev_timestamp = timestamp
            prev_emotion = emotion
        
        # Make sure we're using string keys in all dictionaries for JSON serialization
        return {
            "dominant_emotion": self.get_dominant_emotion(),
            "emotion_sequence": self.emotion_history,
            "transitions": dict(self.transition_counts),  # Now using string keys
            "most_frequent_transition": self.get_most_frequent_transition(),
            "recent_trend": self.get_recent_trend(),
            "emotion_durations": emotion_durations
        }


class EmotionAwareConversationSystem:
    def __init__(self):
        self.conversation_history = []
        self.user_emotions = []
        self.current_strategy = None
        self.emotion_tracker = EmotionTracker()
        self.conversation_turns = 0
    
    def encode_audio(self, audio_file_path: str) -> str:
        """Encode an audio file to base64 string"""
        if audio_file_path.startswith("http"):
            # Download from URL
            response = requests.get(audio_file_path)
            response.raise_for_status()
            audio_data = response.content
        else:
            # Read local file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
        
        return base64.b64encode(audio_data).decode('utf-8')
    
    def detect_emotion(self, audio_base64: str) -> str:
        """Detect emotion from audio using the OpenAI API"""
        print("Detecting emotion...")
        
        try:
            # FIX 1: Add audio output configuration
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What emotion is expressed in this audio?"},
                        {"type": "audio", "audio": audio_base64}
                    ]}
                ],
                # Add this output configuration to fix the error
                output="text"  # This tells the API we just want text output, not audio
            )
            
            content = response.choices[0].message.content
            
            # Extract emotion from response
            for emotion in EMOTION_STATES:
                if emotion.lower() in content.lower():
                    print(f"Detected emotion: {emotion}")
                    return emotion
                    
            # Default to neutral if no clear emotion is detected
            print("Detected emotion: neutral (default)")
            return "neutral"
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return "neutral"  # Default to neutral on error
    
    def select_response_strategy(self, current_emotion: str, previous_emotion: Optional[str] = None) -> Dict:
        """Select appropriate response strategy based on emotion or emotion transition"""
        if previous_emotion is None or previous_emotion == current_emotion:
            # Use single emotion strategy
            return RESPONSE_STRATEGIES[current_emotion]
        else:
            # Check for a specific transition strategy
            transition_key = (previous_emotion, current_emotion)
            if transition_key in TRANSITION_STRATEGIES:
                return TRANSITION_STRATEGIES[transition_key]
            else:
                # Default to the strategy for current emotion if no transition strategy exists
                return RESPONSE_STRATEGIES[current_emotion]
    
    def generate_response(self, user_input: str, detected_emotion: str, strategy: Dict, conversation_history=None) -> str:
        """Generate response based on user input, emotion, and selected strategy"""
        print("Generating response...")
        
        try:
            # FIX 1: Add proper audio output configuration
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an empathetic assistant that responds with awareness of the user's emotional state. The user seems {detected_emotion}. Use the following response strategy: {strategy['name']} - {strategy['description']}"},
                    {"role": "user", "content": user_input}
                ],
                # Remove audio output configuration or set it properly if needed
                output="text"  # We only want text output here
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I'm sorry, I wasn't able to process your message properly. Could you please try again?"
    
    def process_conversation_turn(self, audio_file_path: str) -> str:
        """Process a complete conversation turn with emotion detection and response generation"""
        self.conversation_turns += 1
        print(f"\n=== Processing Conversation Turn {self.conversation_turns} ===")
        
        # 1. Encode audio
        print("Encoding audio...")
        audio_data = self.encode_audio(audio_file_path)
        
        # 2. Detect emotion
        print("Detecting emotion...")
        current_emotion = self.detect_emotion(audio_data)
        print(f"Detected emotion: {current_emotion}")
        
        # 3. Update emotion tracker
        self.emotion_tracker.add_emotion(current_emotion)
        
        # 4. Determine previous emotion (if any)
        previous_emotion = self.user_emotions[-1] if self.user_emotions else None
        
        # 5. Store emotion in history
        self.user_emotions.append(current_emotion)
        
        # 6. Select response strategy
        print("Selecting response strategy...")
        strategy = self.select_response_strategy(current_emotion, previous_emotion)
        self.current_strategy = strategy
        print(f"Selected strategy: {strategy['name']} - {strategy['description']}")
        
        # 7. Generate response
        print("Generating response...")
        response = self.generate_response(audio_data, current_emotion, strategy, self.conversation_history)
        
        # 8. Store in conversation history
        self.conversation_history.append({
            "turn": self.conversation_turns,
            "user_emotion": current_emotion,
            "strategy_used": strategy['name'],
            "response": response
        })
        
        # 9. Print emotion transition information if applicable
        if previous_emotion and previous_emotion != current_emotion:
            print(f"\nEmotion Transition Detected: {previous_emotion} → {current_emotion}")
            
            # Get trend information
            trend = self.emotion_tracker.get_recent_trend()
            print(f"Recent emotion trend: {trend}")
        
        return response
    
    def get_conversation_metrics(self) -> Dict:
        """Generate metrics about the conversation"""
        if not self.conversation_history:
            return {"status": "No conversation data available"}
        
        # Calculate basic metrics
        emotion_changes = sum(1 for i in range(1, len(self.user_emotions)) 
                             if self.user_emotions[i] != self.user_emotions[i-1])
        
        # Get transition report from emotion tracker
        transition_report = self.emotion_tracker.generate_transition_report()
        
        # Calculate strategy distribution
        strategy_counts = {}
        for turn in self.conversation_history:
            strategy = turn["strategy_used"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "conversation_summary": {
                "turns": len(self.conversation_history),
                "emotion_changes": emotion_changes,
                "emotions_detected": self.user_emotions,
                "strategies_used": [turn["strategy_used"] for turn in self.conversation_history]
            },
            "emotion_analysis": transition_report,
            "strategy_distribution": strategy_counts
        }
    
    def evaluate_system_performance(self) -> Dict:
        """Evaluate system performance based on various metrics"""
        # In a real system, these would be calculated against ground truth or user feedback
        if not self.conversation_history:
            return {"status": "No conversation data available for evaluation"}
        
        # Get emotion transition report
        transition_report = self.emotion_tracker.generate_transition_report()
        
        # Example metrics - in a real system these would be calculated from actual data
        strategy_appropriateness = 0.85  # Example value
        
        evaluation = {
            "emotion_recognition": {
                "accuracy": 0.92,  # Would be calculated against ground truth in a real system
                "confidence": 0.88
            },
            "emotion_transitions": {
                "detection_accuracy": 0.90,
                "transition_patterns": transition_report["transitions"]
            },
            "response_strategies": {
                "appropriateness": strategy_appropriateness,
                "consistency": 0.95
            },
            "overall_quality": {
                "coherence": 0.91,
                "empathy": 0.87,
                "user_satisfaction": 0.82
            }
        }
        
        return evaluation


def simulate_emotion_changes(conversation_system):
    """Simulate a conversation with emotion changes to test the system"""
    # Example audio files with different emotions
    # In a real implementation, these would be actual different audio files
    neutral_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"  # Pretend this is neutral
    happy_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"    # Pretend this is happy
    angry_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"    # Pretend this is angry
    sad_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"      # Pretend this is sad
    
    # Force emotion detection for simulation purposes
    original_detect_emotion = conversation_system.detect_emotion
    
    # Turn 1: Neutral
    print("\n=== SIMULATING CONVERSATION WITH EMOTION CHANGES ===")
    conversation_system.detect_emotion = lambda x: "neutral"
    response = conversation_system.process_conversation_turn(neutral_audio)
    print(f"\nSystem Response (Neutral):\n{response}\n")
    
    # Turn 2: Happy
    conversation_system.detect_emotion = lambda x: "happy"
    response = conversation_system.process_conversation_turn(happy_audio)
    print(f"\nSystem Response (Happy):\n{response}\n")
    
    # Turn 3: Angry (emotion change)
    conversation_system.detect_emotion = lambda x: "angry"
    response = conversation_system.process_conversation_turn(angry_audio)
    print(f"\nSystem Response (Angry):\n{response}\n")
    
    # Turn 4: Still angry (no change)
    response = conversation_system.process_conversation_turn(angry_audio)
    print(f"\nSystem Response (Still Angry):\n{response}\n")
    
    # Turn 5: Sad (emotion change)
    conversation_system.detect_emotion = lambda x: "sad"
    response = conversation_system.process_conversation_turn(sad_audio)
    print(f"\nSystem Response (Sad):\n{response}\n")
    
    # Restore original function
    conversation_system.detect_emotion = original_detect_emotion
    
    # Print detailed metrics
    metrics = conversation_system.get_conversation_metrics()
    print("\n=== CONVERSATION METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    # Print system evaluation
    evaluation = conversation_system.evaluate_system_performance()
    print("\n=== SYSTEM EVALUATION ===")
    print(json.dumps(evaluation, indent=2))


def main():
    """Main function to demonstrate the emotion-aware conversation system"""
    print("Initializing Emotion-Aware Conversation System...")
    conversation_system = EmotionAwareConversationSystem()
    
    # Option 1: Real usage with actual audio input
    use_real_audio = False  # Set to True to use real audio processing
    
    if use_real_audio:
        # First turn
        print("Processing first conversation turn...")
        audio_file = "https://cdn.openai.com/API/docs/audio/alloy.wav"  # Example audio
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Subsequent turn with potentially different emotion
        print("Processing second conversation turn...")
        # In a real application, this would be a new audio file
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Print conversation metrics
        print("Conversation Metrics:")
        print(json.dumps(conversation_system.get_conversation_metrics(), indent=2))
    
    else:
        # Option 2: Simulation with forced emotion changes to demonstrate the system
        simulate_emotion_changes(conversation_system)


if __name__ == "__main__":
    main()

    conversation_system = EmotionAwareConversationSystem()
    
    # Option 1: Real usage with actual audio input
    use_real_audio = False  # Set to True to use real audio processing
    
    if use_real_audio:
        # First turn
        print("Processing first conversation turn...")
        audio_file = "https://cdn.openai.com/API/docs/audio/alloy.wav"  # Example audio
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Subsequent turn with potentially different emotion
        print("Processing second conversation turn...")
        # In a real application, this would be a new audio file
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Print conversation metrics
        print("Conversation Metrics:")
        print(json.dumps(conversation_system.get_conversation_metrics(), indent=2))
    
    else:
        # Option 2: Simulation with forced emotion changes to demonstrate the system
        simulate_emotion_changes(conversation_system)


if __name__ == "__main__":
    main()

