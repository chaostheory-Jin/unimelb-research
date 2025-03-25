import json

def evaluate_system_performance(conversation_system):
    """Evaluate system performance based on various metrics"""
    # Get metrics from conversation system
    metrics = conversation_system.get_conversation_metrics()
    
    # Get emotion transition report
    transition_report = conversation_system.emotion_tracker.generate_transition_report()
    
    # Calculate strategy appropriateness
    # In a real system, this would be based on human evaluations or predefined rules
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

def evaluate_emotion_transitions():
    """Evaluate how the system handles emotion transitions"""
    print("\n=== EVALUATING EMOTION TRANSITION HANDLING ===\n")
    
    conversation_system = EmotionAwareConversationSystem()
    
    # Key emotion transition sequences
    transitions = [
        ("neutral", "angry"),  # Neutral to angry
        ("happy", "sad"),      # Happy to sad
        ("angry", "happy"),    # Angry to happy
        ("sad", "anxious"),    # Sad to anxious
    ]
    
    results = {}
    
    # Test each transition
    for prev_emotion, curr_emotion in transitions:
        print(f"\n--- Testing transition: {prev_emotion} → {curr_emotion} ---")
        
        # Reset system
        conversation_system = EmotionAwareConversationSystem()
        
        # First process with first emotion
        conversation_system.process_conversation_turn(emotion_override=prev_emotion)
        
        # Then process with second emotion
        response = conversation_system.process_conversation_turn(emotion_override=curr_emotion)
        
        # Get selected strategy
        strategy = conversation_system.conversation_history[-1]["strategy"]
        
        # Record results
        transition_key = f"{prev_emotion}_to_{curr_emotion}"
        results[transition_key] = {
            "strategy": strategy,
            "response": response
        }
        
        print(f"Selected strategy: {strategy}")
        print(f"Response: {response[:100]}...")  # Only show first 100 chars of response
    
    # Output summary of results
    print("\n=== TRANSITION HANDLING SUMMARY ===")
    for transition, data in results.items():
        print(f"{transition}: Used strategy '{data['strategy']}'")
    
    return results

def evaluate_with_simulated_emotions():
    """Evaluate the system using a simulated sequence of emotions"""
    print("\n=== EVALUATING SYSTEM WITH SIMULATED EMOTIONS ===\n")
    
    conversation_system = EmotionAwareConversationSystem()
    
    # Define sequence of emotions to test
    emotion_sequence = ["neutral", "happy", "angry", "sad", "anxious", "happy"]
    
    # Track responses and strategies
    responses = []
    strategies = []
    
    # Run a conversation turn for each emotion
    for emotion in emotion_sequence:
        print(f"\n--- Testing emotion: {emotion} ---")
        response = conversation_system.process_conversation_turn(emotion_override=emotion)
        responses.append(response)
        strategies.append(conversation_system.current_strategy)
    
    # Output evaluation metrics
    print("\n=== EVALUATION METRICS ===")
    print(f"Conversation turns successfully processed: {len(responses)}")
    print(f"Number of emotion transitions: {len(conversation_system.emotion_tracker.transition_counts)}")
    
    # Check if strategies were appropriate for emotions
    unique_strategies = set([strategy['name'] for strategy in conversation_system.conversation_history])
    print(f"Number of unique strategies used: {len(unique_strategies)}")
    
    # Output emotion report
    emotion_report = conversation_system.emotion_tracker.generate_transition_report()
    print("\nEmotion Tracking Report:")
    print(json.dumps(emotion_report, indent=2))
    
    return emotion_report

def evaluate_with_crema_d_examples():
    """Evaluate the system using CREMA-D dataset samples"""
    print("\n=== EVALUATING SYSTEM WITH CREMA-D SAMPLES ===\n")
    
    conversation_system = EmotionAwareConversationSystem()
    
    # List of example audio file paths (adjust according to your actual paths)
    audio_samples = [
        "path/to/crema-d/1001_IEO_ANG_XX.wav",  # Angry sample
        "path/to/crema-d/1001_IEO_HAP_XX.wav",  # Happy sample
        "path/to/crema-d/1001_IEO_SAD_XX.wav",  # Sad sample
        "path/to/crema-d/1001_IEO_NEU_XX.wav",  # Neutral sample
    ]
    
    # Expected emotion labels (based on filenames)
    expected_emotions = ["angry", "happy", "sad", "neutral"]
    
    # Track detection accuracy
    correct_detections = 0
    
    # Run a conversation turn for each audio sample
    for i, audio_file in enumerate(audio_samples):
        print(f"\n--- Testing sample: {audio_file} ---")
        response = conversation_system.process_conversation_turn(audio_data=audio_file)
        
        # Check if detected emotion matches expected
        detected_emotion = conversation_system.conversation_history[-1]["emotion"]
        if detected_emotion == expected_emotions[i]:
            correct_detections += 1
            print(f"✓ Emotion correctly detected: expected={expected_emotions[i]}, detected={detected_emotion}")
        else:
            print(f"✗ Emotion mismatch: expected={expected_emotions[i]}, detected={detected_emotion}")
    
    # Output evaluation metrics
    print("\n=== EVALUATION METRICS ===")
    print(f"Total samples: {len(audio_samples)}")
    print(f"Correct detections: {correct_detections}")
    print(f"Detection accuracy: {correct_detections/len(audio_samples)*100:.2f}%")
    
    # Output emotion report
    emotion_report = conversation_system.emotion_tracker.generate_transition_report()
    print("\nEmotion Tracking Report:")
    print(json.dumps(emotion_report, indent=2))
    
    return emotion_report

def evaluate_with_existing_dialogue():
    """Evaluate the system using predefined dialogue"""
    print("\n=== EVALUATING SYSTEM WITH PREDEFINED DIALOGUE ===\n")
    
    conversation_system = EmotionAwareConversationSystem()
    
    # Predefined dialogue samples (user emotion, user input)
    dialogue = [
        ("neutral", "How's the weather today?"),
        ("happy", "I just passed my exam, I'm so excited!"),
        ("angry", "Why has my account been locked? This is ridiculous!"),
        ("sad", "I'm feeling down, life is hard..."),
        ("anxious", "I have an important presentation tomorrow, I'm nervous")
    ]
    
    for emotion, text in dialogue:
        print(f"\n--- User: {text} (Emotion: {emotion}) ---")
        # Pretend we have audio with emotion, but use emotion override
        response = conversation_system.process_conversation_turn(emotion_override=emotion)
    
    # Output emotion report
    emotion_report = conversation_system.emotion_tracker.generate_transition_report()
    print("\nEmotion Tracking Report:")
    print(json.dumps(emotion_report, indent=2))
    
    return emotion_report

def main():
    print("Initializing Emotion-Aware Conversation System...")
    
    # 1. Evaluate with simulated emotions
    print("\n\n========= METHOD 1: SIMULATED EMOTION EVALUATION =========")
    simulate_results = evaluate_with_simulated_emotions()
    
    # 2. Evaluate with CREMA-D samples if real audio is available
    use_real_audio = False  # Set to True to use real audio
    if use_real_audio:
        print("\n\n========= METHOD 2: CREMA-D SAMPLE EVALUATION =========")
        crema_results = evaluate_with_crema_d_examples()
    
    # 3. Evaluate emotion transition handling
    print("\n\n========= METHOD 3: EMOTION TRANSITION EVALUATION =========")
    transition_results = evaluate_emotion_transitions()
    
    print("\n\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
    # Or evaluate separately
    # evaluate_with_existing_dialogue()
