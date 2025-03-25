# Additional code for enhanced emotion tracking

class EmotionTracker:
    def __init__(self):
        self.emotion_history = []
        self.transition_counts = {}
        
    def add_emotion(self, emotion: str):
        self.emotion_history.append(emotion)
        
        # Track transitions
        if len(self.emotion_history) > 1:
            prev = self.emotion_history[-2]
            curr = self.emotion_history[-1]
            transition = (prev, curr)
            
            if transition in self.transition_counts:
                self.transition_counts[transition] += 1
            else:
                self.transition_counts[transition] = 1
    
    def get_most_frequent_transition(self) -> Tuple[Tuple[str, str], int]:
        if not self.transition_counts:
            return (("none", "none"), 0)
        
        most_freq = max(self.transition_counts.items(), key=lambda x: x[1])
        return most_freq
    
    def get_dominant_emotion(self) -> str:
        if not self.emotion_history:
            return "none"
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def generate_transition_report(self) -> Dict:
        return {
            "dominant_emotion": self.get_dominant_emotion(),
            "emotion_sequence": self.emotion_history,
            "transitions": dict(self.transition_counts),
            "most_frequent_transition": self.get_most_frequent_transition()
        }


# Add to EmotionAwareConversationSystem class
def __init__(self):
    self.conversation_history = []
    self.user_emotions = []
    self.current_strategy = None
    self.emotion_tracker = EmotionTracker()
    
# Update process_conversation_turn method
def process_conversation_turn(self, audio_file_path: str) -> str:
    # ... existing code
    
    # After emotion detection
    current_emotion = self.detect_emotion(audio_data)
    print(f"Detected emotion: {current_emotion}")
    
    # Update emotion tracker
    self.emotion_tracker.add_emotion(current_emotion)
    
    # ... remaining code
