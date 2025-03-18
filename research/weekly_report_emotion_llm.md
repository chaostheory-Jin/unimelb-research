# Weekly Research Report: Emotion-Understanding Audio-Text-to-Text Model

## Project Overview

This project aims to fine-tune an audio-text-to-text large language model to better understand human emotions and improve human-computer dialogue systems accordingly. The current research phase focuses on literature review, available dataset analysis, and task design.

## Research Progress: Literature Review

### Overview of Audio-Text-to-Text Models

Audio-text-to-text models are multimodal large language models capable of processing both audio and text inputs and generating text outputs. These models typically consist of the following components:

1. **Audio Encoder**: Converts audio signals into vector representations
2. **Text Encoder**: Processes text inputs and generates text vector representations
3. **Multimodal Fusion Module**: Integrates audio and text information
4. **Text Decoder**: Generates responses based on the fused representations

Representative models include:
- **Whisper-LLM**: Based on OpenAI's Whisper model combined with LLM
- **BLSP (Better Long Short Prompting)**: Focused on long text and audio interaction
- **Qwen-Audio**: Alibaba's multimodal audio-text model

## Analysis of Available Emotion Datasets

### 1. IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)

- **Source**: Created by USC
- **Format**: Dialogue audio files (.wav) + emotion annotation files (.txt)
- **Emotion Categories**: Anger, happiness, sadness, neutral, excitement, frustration
- **Scale**: Approximately 12 hours of dialogue, 10 actors
- **Features**: Includes facial expressions, audio, and text transcriptions
- **License**: Available upon application for research purposes

### 2. MELD (Multimodal EmotionLines Dataset)

- **Source**: Based on the TV series "Friends"
- **Format**: Dialogue video/audio clips + text transcriptions + emotion labels
- **Emotion Categories**: Anger, disgust, fear, joy, neutral, sadness, surprise
- **Scale**: Approximately 13,000 utterances, 1,433 dialogues
- **Features**: Multi-turn dialogues with context
- **License**: Open access

### 3. MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)

- **Source**: YouTube video collection created by CMU
- **Format**: Video/audio clips + text transcriptions + emotion intensity labels
- **Emotion Categories**: Anger, disgust, fear, happiness, sadness, surprise (with intensity scores)
- **Scale**: 23,500 sentences, over 3,000 videos
- **Features**: Includes emotion intensity ratings
- **License**: Open for research purposes

### 4. EmoWOZ

- **Source**: Emotion-extended version of the MultiWOZ dataset
- **Format**: Dialogue text + emotion labels + task labels
- **Emotion Categories**: Positive, negative, neutral
- **Scale**: Approximately 10,000 multi-turn dialogues
- **Features**: Emotion annotations in task-oriented dialogues
- **License**: Open access

### 5. GoEmotions

- **Source**: Text emotion dataset developed by Google
- **Format**: Text + multi-label emotions
- **Emotion Categories**: 27 fine-grained emotions and sentiments
- **Scale**: 58,000 Reddit comments
- **Features**: Fine-grained emotion classification
- **License**: Open access (CC BY 4.0)

## Data Format Analysis

Most emotion datasets adopt one of the following formats:

1. **Standard Format**:
   ```
   {
     "id": "dialogue_123",
     "utterances": [
       {
         "speaker": "A",
         "text": "How was your day?",
         "audio": "path/to/audio1.wav",
         "emotion": "neutral"
       },
       {
         "speaker": "B",
         "text": "Terrible, my project ran into issues.",
         "audio": "path/to/audio2.wav",
         "emotion": "frustrated"
       }
     ]
   }
   ```

2. **Temporal Annotation Format**:
   ```
   {
     "id": "utterance_456",
     "audio": "path/to/audio.wav",
     "text": "I really love this idea!",
     "start_time": 10.5,
     "end_time": 13.2,
     "emotion": "excited",
     "intensity": 0.85
   }
   ```

## Dialogue Improvement Task Ideas

Based on the model's understanding of emotions, the following tasks can be designed to improve dialogue systems:

### 1. Emotion-Sensitive Response Generation

- **Task Description**: Adjust the tone and content of responses based on detected user emotions
- **Example**: When the user is detected to be sad, the model generates gentler, more sympathetic responses

### 2. Emotional State Tracking

- **Task Description**: Continuously track changes in user emotional states during multi-turn dialogues
- **Example**: The model remembers the user's previously expressed disappointment and reflects understanding and support in subsequent responses

#### Expanded Description of Emotional State Tracking

Emotional State Tracking involves building a dynamic emotional memory mechanism into dialogue systems that can:

1. **Emotional History Maintenance**: Create and maintain an emotional trajectory of the user throughout the conversation, recording not just the current emotional state but also previous states and the transitions between them.

2. **Contextual Understanding**: Interpret new utterances in the context of this emotional history, rather than treating each turn in isolation. For example, a neutral statement following anger might be interpreted differently than the same statement in a consistently neutral conversation.

3. **Intensity Tracking**: Monitor not just the category of emotion but also its intensity and how it changes over time (e.g., gradually diminishing anger vs. sudden shifts).

4. **Causal Attribution**: Identify and record potential causes of emotional shifts based on conversation content, which can inform more targeted responses.

5. **Long-term Memory**: Maintain emotional profiles across multiple conversations, enabling the model to remember a user's emotional tendencies and previous emotional experiences with certain topics.

**Implementation Approaches**:
- Graph-based representations of emotional states and transitions
- Recurrent neural architectures with explicit emotional memory cells
- Transformer-based approaches with specialized attention mechanisms for emotional content
- Hybrid models that combine rule-based emotional state machines with neural understanding

**Evaluation Metrics**:
- Emotion tracking accuracy over time (compared to human annotations)
- Appropriateness of responses given the emotional history
- User satisfaction with the system's emotional memory capabilities
- Ability to recall and reference previous emotional states at appropriate times

### 3. Emotion Transition Strategies

- **Task Description**: Design dialogue strategies to help users transition from negative to positive emotions
- **Example**: When a user appears dejected, the model can first express understanding, then gradually guide the topic towards a positive direction

#### Expanded Description of Emotion Transition Strategies

Emotion Transition Strategies involve developing systematic approaches to guide users from negative emotional states to more positive ones through carefully designed conversation flows. This includes:

1. **Emotional Validation Phase**: First acknowledging and legitimizing the user's negative emotions without immediately attempting to change them:
   - "I understand you're feeling frustrated about this situation."
   - "It makes sense that you'd feel disappointed after what happened."

2. **Bridging Techniques**: Employing specific linguistic and psychological techniques to create natural transitions:
   - Perspective shifting: "While this aspect was challenging, how do you feel about...?"
   - Subtle reframing: "I wonder if we might think about this challenge as an opportunity to..."
   - Strategic questioning: Asking questions that gradually direct attention to more positive aspects

3. **Positive Engagement Strategies**: Introducing topics or questions designed to evoke positive emotional responses:
   - Gratitude elicitation: "Despite these difficulties, is there anything you've appreciated lately?"
   - Achievement focus: "What's something you've managed well recently, even if small?"
   - Future orientation: "How would you like things to be different moving forward?"

4. **Personalized Approach Mapping**: Developing different transition strategies based on:
   - The specific negative emotion being experienced (anger vs. sadness vs. anxiety)
   - The intensity of the emotion (mild disappointment vs. severe distress)
   - The user's past responsiveness to different transition techniques
   - Cultural factors that influence emotional expression and regulation

5. **Ethical Considerations**: Ensuring transitions are:
   - Non-manipulative and transparent
   - Respectful of the user's autonomy
   - Appropriate to the situation (not dismissing genuinely serious concerns)
   - Supporting healthy emotional processing rather than suppression

**Implementation Approaches**:
- Decision tree frameworks for selecting appropriate transition strategies
- Reinforcement learning to optimize transition success over time
- Sequence-to-sequence models trained specifically on emotional transition dialogues
- Template-based approaches with dynamic content selection

**Evaluation Metrics**:
- Pre/post emotional state measurements
- User-reported helpfulness of conversations
- Naturalness of transitions (as rated by human evaluators)
- Long-term emotional impact across multiple conversations

### 4. Multimodal Emotion Consistency

- **Task Description**: Ensure the model generates responses with consistent emotions in both text content and audio features (if voice output is available)
- **Example**: Ensuring sympathetic content is not paired with fast, high-pitched tones

### 5. Culturally Sensitive Emotion Understanding

- **Task Description**: Consider differences in emotional expression across different cultural backgrounds
- **Example**: Adjust understanding and response methods for specific emotional signals based on the user's cultural background

## Next Steps

1. In-depth research on MELD and GoEmotions datasets to evaluate their applicability
2. Design specific task descriptions and evaluation methods for emotion-sensitive response generation
3. Explore optimal fusion methods for joint audio-text representation
4. Develop fine-tuning strategies and experimental plans

## References

1. Colombo, P., et al. (2022). "Affect-driven dialog generation."
2. Poria, S., et al. (2019). "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations."
3. Zadeh, A., et al. (2018). "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph."
4. Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." 