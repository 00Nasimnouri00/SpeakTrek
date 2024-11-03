#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from telebot import TeleBot
from dotenv import load_dotenv
from pydub import AudioSegment
import ffmpeg
import requests
import assemblyai as aai
import re


# In[ ]:


def emotion_important(text_segm):
    url = 'https://api.metisai.ir/api/v1/wrapper/openai_chat_completion/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {os.getenv('API_AUTHORIZATION_TOKEN')}"
    }
    data = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant trained to identify emotionally charged phrases in a given text."
            },
            {
                "role": "user",
                "content": (
                    "Given a segment of text, identify and list emotionally charged phrases. Focus on phrases that convey emotion, visualization, or introspection. "
                    "Return the results as a list, separating each phrase that could evoke an emotional response."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Text: \"{text_segm}\"\n\n"
                    "Now, analyze the following text:"
                )
            }
        ]
    }
    
    Res = requests.post(url, headers=headers, json=data)
    return Res


##########################
def convert_to_list(text):
    lines = text.splitlines()[1:]  # Skip the first line "Emotionally charged phrases:\n"
    phrases = [line.strip('- ').strip() for line in lines if line.strip()]
    return phrases

def match_emotion_pitches(phrase_list, pitch_values):
    matched_pitches = {}
    current_index = 0
    
    for phrase in phrase_list:
        words = phrase.split()
        phrase_length = len(words)
        
        if current_index + phrase_length <= len(pitch_values):
            matched_pitches[phrase] = pitch_values[current_index:current_index + phrase_length]
            current_index += phrase_length
        else:
            break
    
    return matched_pitches

# Function to handle emotion and pitch analysis for segments
def analyze_emotion_and_pitch(segment_pitches, message):
    emotion_list = []
    matched_pitched_list = []

    for segment_data in segment_pitches:
        text = segment_data['text']
        segment_pitch_values = segment_data['pitches']
        
        response = emotion_important(text)
        
        # Check if response has 'status_code' and if it is 200
        if hasattr(response, 'status_code') and response.status_code == 200:
            response_data = response.json()
            text_content = response_data["choices"][0]['message']['content']
            
            emotionally_significant_phrases = convert_to_list(text_content)
            emotion_list.append(emotionally_significant_phrases)
            
            matched_emotion_pitches = match_emotion_pitches(emotionally_significant_phrases, segment_pitch_values)
            matched_pitched = [matched_emotion_pitches[phrase] for phrase in emotionally_significant_phrases]
            matched_pitched_list.append(matched_pitched)
        else:
            error_msg = response.get('message', 'An error occurred')
            bot.reply_to(message, f"Error: {response.get('status_code', 'Unknown Status')}. {error_msg}")
            return None, None  # Return None values in case of an error
    
    bot.reply_to(message, "analyze_emotion_and_pitch is done")
    
    # Return emotion_list and matched_pitched_list
    return emotion_list, matched_pitched_list


############################################

def transcribe_plot_extract_pitches(audio_file, message):
    try:
        ##VOICE PREPARATION##################################################################################################
        # Load audio for pitch analysis
        y, sr = librosa.load(audio_file)
        
        # Extract pitches and magnitudes
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, hop_length=512)

        # Process to find F0 (pitch) over time
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch if pitch > 0 else np.nan)  # Filter out zero or undefined pitches

        # Plot the pitch variations over time
        plt.figure(figsize=(14, 5))
        plt.plot(pitch_values, label='Pitch (F0)')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Pitch Variation in Speech')
        plt.legend()

        # Save the plot as an image file
        plot_path = "pitch_plot.png"
        plt.savefig(plot_path)
        plt.close()

        # Send the plot to the user
        with open(plot_path, 'rb') as plot_file:
            bot.send_photo(message.chat.id, plot_file)

        # Clean up the image file
        os.remove(plot_path)


        ##text PREPARATION##################################################################################################
        aai.settings.api_key = "9084b5d030f0484c93774461752dcaca"
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)
        # transcript = transcriber.transcribe("./my-local-audio-file.wav")
        
        sentences = re.split(r'(?<=[.!?]) +', transcript.text)
        
        words_frame=[]
        words_text=[]
        for i in range(len(transcript.words)):
            char={}
            char["start"]=transcript.words[i].start/1000
            char["end"]=transcript.words[i].end/1000
            words_frame.append(char)
            words_text.append(transcript.words[i])
        
        segmented_text = []  # List to hold words for each sentence
        stert_end_segmented_text=[]
        
        n = 0   # Sentence index
        Sent = sentences
        current_sentence_words = []  # Temporary list for words in the current sentence
        current_sentence_pitch=[]
        # Iterate over each word
        for c in words_text:
            # Append word to the current sentence list
            current_sentence_words.append(c.text)
            current_sentence_pitch.append(c)
            
            # Check if the word completes the current sentence
            if all(word in ' '.join(current_sentence_words) for word in Sent[n].split()):
                # If the sentence is complete, add to the main list and reset
                segmented_text.append(current_sentence_words)
                stert_end_segmented_text.append([current_sentence_pitch[0].start/1000,current_sentence_pitch[-1].end/1000])
                
                
                current_sentence_words = []  # Reset for the next sentence
                
                # Move to the next sentence, if any
                if n + 1 < len(Sent):
                    n += 1
                else:
                    break

        ##Audio & text segmentation##################################################################################################
        # Define the segment_pitches for each segment in the transcription result
        hop_length = 512
        # Get pitch values for each segment
        segment_pitches = []
        for i, words_in_sentence in enumerate(segmented_text):
            segment_start = stert_end_segmented_text[i][0]
            segment_end = stert_end_segmented_text[i][1]
            segment_text = words_in_sentence
            
            # Convert times to frame indices
            start_frame = int(segment_start * sr / hop_length)
            end_frame = int(segment_end * sr / hop_length)
            
            # Extract pitch values within these frames
            segment_pitch_values = pitch_values[start_frame:end_frame]
            
            # Store results
            segment_pitches.append({
                "text": segment_text,
                "start": segment_start,
                "end": segment_end,
                "pitches": segment_pitch_values
            })

        # Format and send the transcription and pitch details to the user
        transcription_text = "\n\n".join([
            f"Text: {segment['text']}"
            for segment in segment_pitches
        ])
        bot.reply_to(message, f"Transcription and Pitch Analysis:\n\n{transcription_text}")

        # Return segment_pitches
        return segment_pitches  # Returning the segment pitches

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")
        return None  # Return None in case of an error

##################################################################################
def analyze_text_word(words, related_pitch):
    url = 'https://api.metisai.ir/api/v1/wrapper/openai_chat_completion/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {os.getenv('API_AUTHORIZATION_TOKEN')}"
    }
    data = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant trained to analyze the correlation between pitch and emotional delivery in spoken language."
            },
            {
                "role": "user",
                "content": (
                    "I have identified a set of emotionally charged phrases from a segment of text along with their corresponding pitch values. "
                    "Consider the following:\n\n"
                    "1. **Emotionally Charged Phrases:**\n"
                    + "\n".join(f"   - {phrase}" for phrase in words) +
                    "\n\n2. **Corresponding Pitches:**\n" +
                    "\n".join(f"   - {pitch}" for pitch in related_pitch) +
                    "\n\n"
                    "Your task is to analyze each phrase's emotional delivery based on pitch, and return:\n\n"
                    "1. A score of 'Perfect,' 'Good,' or 'Weak' for each phrase, indicating how well the pitch aligns with its emotional tone.\n"
                    "   - 'Perfect': Pitch fully aligns with the emotional tone.\n"
                    "   - 'Good': Pitch generally aligns but could be improved slightly.\n"
                    "   - 'Weak': Pitch does not align well with the emotional tone.\n\n"
                    "2. If the score is Weak, give practical advice for improving emotional delivery, with specific, concise tips.\n\n"
                )
            }
        ]
    }

    Res2 = requests.post(url, headers=headers, json=data)
    return Res2

####################################################################
# Function to format and send comparison results to the user
def send_comparison_results(comparison_list, message):
    try:
        # Create a detailed response for each phrase analysis
        response_text = "Here is the analysis of your emotionally charged phrases and their pitch alignment:\n\n"
        
        for i, comparison in enumerate(comparison_list, 1):
            response_text += f"**Phrase Analysis {i}:**\n{comparison}\n\n"
            
            # Telegram has a limit of 4096 characters per message, so send in chunks if needed
            if len(response_text) > 3500:  # Send before hitting the limit
                bot.send_message(message.chat.id, response_text)
                response_text = ""  # Reset for the next chunk

        # Send any remaining text that hasn’t been sent yet
        if response_text:
            bot.send_message(message.chat.id, response_text)

    except Exception as e:
        bot.reply_to(message, f"An error occurred while sending analysis results: {str(e)}")


##########################################################################
def calculate_total_score(analysis_text):
    # Initialize total score
    total_score = 0
    
    # Find all score entries in the text
    scores = re.findall(r"\*\*Score:\*\* (Perfect|Good|Weak)", analysis_text)
    
    # Calculate the score for this segment
    segment_score = sum(score_mapping[score] for score in scores)
    total_score += segment_score
    
    return total_score
#########################################################################
# Load environment variables from the .env file
load_dotenv()
bot = TeleBot(os.environ.get("TELEGRAM_BOT_TOKEN"))

# Initialize a dictionary to store each user's position in the comparison list
user_progress = {}

@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me an audio file recorded in Telegram.")

@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio(message):
    try:
        # Notify user that the voice message was received
        bot.reply_to(message, "We received your voice message.")

        # Get the file ID of the audio message
        file_id = message.audio.file_id if message.content_type == 'audio' else message.voice.file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Save the OGG file temporarily
        ogg_path = "user_audio.ogg"
        with open(ogg_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Notify user that the conversion is in progress
        bot.reply_to(message, "Converting audio to .wav format...")

        # Convert OGG to WAV
        wav_path = "user_audio.wav"
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format="wav")
        os.remove(ogg_path)  # Clean up the OGG file

        # Notify user that the plot will be generated
        bot.reply_to(message, "We will send you a plot of the pitch variation.")

        # Analyze the audio and plot pitch
        pitch_seg = transcribe_plot_extract_pitches(wav_path, message)
        emotion_l, match_pitch_l = analyze_emotion_and_pitch(pitch_seg, message)

        # Generate the comparison list
        comparison_list = []
        total_user_score = 0 
        for j in range(len(emotion_l)):
            emotion = emotion_l[j]
            corresponding_pitches = match_pitch_l[j]
            Final_res = analyze_text_word(emotion, corresponding_pitches).json()
            comparison_list.append(Final_res['choices'][0]['message']['content'])
            
            # Calculate score for this segment and add to total
            segment_score = calculate_total_score(analysis_text)
            total_user_score += segment_score
        
        # Send the score to the user
        bot.send_message(message.chat.id, f"Your total emotional delivery score is: {total_user_score}")

        # Start sending the comparison list in batches of five
        user_id = message.chat.id
        user_progress[user_id] = 0  # Initialize user's progress
        send_comparison_results(comparison_list, message)

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

def send_comparison_results(comparison_list, message):
    user_id = message.chat.id
    start = user_progress.get(user_id, 0)
    end = min(start + 5, len(comparison_list))  # Send up to five items at a time

    # Prepare response text for the current batch
    response_text = "Here is your feedback:\n\n"
    for i in range(start, end):
        response_text += f"**Phrase Analysis {i + 1}:**\n{comparison_list[i]}\n\n"
    
    bot.send_message(user_id, response_text)

    # Update user progress or reset if at the end
    if end < len(comparison_list):
        user_progress[user_id] = end  # Update for the next batch
        bot.send_message(user_id, "Would you like to continue with more feedback? Reply 'yes' to continue.")
    else:
        user_progress[user_id] = 0  # Reset for the next use
        bot.send_message(user_id, "You've reached the end of the feedback list.")

@bot.message_handler(func=lambda message: message.text.lower() == "yes")
def handle_continue_feedback(message):
    user_id = message.chat.id
    # Check if user has more feedback to receive
    if user_id in user_progress and user_progress[user_id] < len(comparison_list):
        send_comparison_results(comparison_list, message)
    else:
        bot.send_message(user_id, "No more feedback to show.")

if __name__ == "__main__":
    bot.polling()



# In[ ]:




