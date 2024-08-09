import pyaudio
import wave
import numpy as np
from elevenlabs import generate, stream
from openai import OpenAI
import time
import os
import keyboard

class AI_Assistant:
    def __init__(self):
        self.openai_client = OpenAI(api_key="OPEN_AI API_KEY") #入力箇所1
        self.elevenlabs_api_key = "ELEVEN_LABS_API_KEY"#入力箇所2
        self.full_transcript = [
            {"role": "system", "content": "あなたはAIアシスタントのワタナベです。機転を聞かせ、効率的に対応してください。すべての応答を完全にひらがなとカタカナのみで行ってください。漢字は一切使用しないでください。"}
        ]
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.continue_conversation = True

    def start_transcription(self):
        self.is_recording = True
        self.frames = []
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        print("聞き取り中... 話し終わったらスペースキーを押してください。")
        
        while self.is_recording:
            if keyboard.is_pressed('space'):
                print("\n録音を停止しています...")
                self.is_recording = False
            else:
                data = self.stream.read(1024)
                self.frames.append(data)

    def stop_transcription(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if len(self.frames) > 0:
            wf = wave.open("temp_audio.wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            with open("temp_audio.wav", "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            os.remove("temp_audio.wav")
            return transcript
        return None

    def generate_ai_response(self, transcript):
        self.full_transcript.append({"role": "user", "content": transcript})
        print(f"\nユーザー: {transcript}")

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.full_transcript
        )

        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)

    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAIアシスタント ワタナベ: {text}")

        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            voice="VOICE ID",  #入力箇所3
            model="eleven_multilingual_v2",  #入力箇所4
            stream=True
        )

        stream(audio_stream)

    def run(self):
        greeting = "こんにちは。ワタナベともうします。何かお手伝いできることはありますか？"
        self.generate_audio(greeting)

        while self.continue_conversation:
            self.start_transcription()
            transcript = self.stop_transcription()
            if transcript:
                self.generate_ai_response(transcript)
            print("\n次の入力の準備ができました。話し終わったらスペースキーを押すか、終了する場合は'q'を押してください。")
            
            if keyboard.is_pressed('q'):
                self.continue_conversation = False

        print("会話が終了しました。ご利用ありがとうございました。")
        self.audio.terminate()

# メイン実行
if __name__ == "__main__":
    ai_assistant = AI_Assistant()
    ai_assistant.run()