import asyncio
import io
import websockets
from pathlib import Path
import wave
from google import genai
from google.genai import types
import soundfile as sf
import librosa
from dotenv import load_dotenv
import os
import os
import sys
import json
import base64
import asyncio
import contextlib
import io
import threading
import time
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from websockets.asyncio.client import connect



# --- 設定 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables")

HOST = 'generativelanguage.googleapis.com'
MODEL = 'models/gemini-live-2.5-flash-preview'
INITIAL_REQUEST_TEXT = "HI" #預設會講的話

# --- Audio 設定 24000、2400 不能更改 ---
@dataclass
class AudioConfig:
    """Configuration for audio streams."""
    sample_rate: int = 24000
    block_size: int = 2400
    channels: int = 1
    dtype: str = 'int16'

# --- Audio Handling Class  ---
class RunningLiveAudio:
    """Manages real-time audio input/output using sounddevice."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue()
        self._output_buffer = io.BytesIO()
        self._output_lock = threading.Lock()
        self._stream = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)

        # Feed microphone data into the asyncio queue
        self._loop.call_soon_threadsafe(self._input_queue.put_nowait, indata.tobytes())

        # Playback audio from the output buffer
        with self._output_lock:
            bytes_to_read = frames * outdata.dtype.itemsize * outdata.shape[1]
            chunk = self._output_buffer.read(bytes_to_read)
            if len(chunk) < bytes_to_read:
                padding_bytes = bytes_to_read - len(chunk)
                chunk += b'\x00' * padding_bytes
                # Reset buffer since it's now empty
                remaining = self._output_buffer.read()
                self._output_buffer.seek(0)
                self._output_buffer.truncate(0)
                self._output_buffer.write(remaining)
                self._output_buffer.seek(0)
            
            np_chunk = np.frombuffer(chunk, dtype=outdata.dtype).reshape(outdata.shape)
            outdata[:] = np_chunk

    async def __aenter__(self):
        print("Starting audio stream...")
        self._stream = sd.Stream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            dtype=self.config.dtype,
            channels=self.config.channels,
            callback=self._audio_callback
        )
        self._stream.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            print("Audio stream stopped.")

    async def read(self):
        """Read a chunk of audio from the microphone."""
        return await self._input_queue.get()

    async def enqueue(self, data: bytes):
        """Enqueue a chunk of audio data for playback."""
        def _write_to_buffer():
            with self._output_lock:
                pos = self._output_buffer.tell()
                self._output_buffer.seek(0, io.SEEK_END)
                self._output_buffer.write(data)
                self._output_buffer.seek(pos)
        await self._loop.run_in_executor(None, _write_to_buffer)

    async def clear_queue(self):
        """Clear the playback queue."""
        def _clear_buffer():
            with self._output_lock:
                self._output_buffer.seek(0)
                self._output_buffer.truncate(0)
        await self._loop.run_in_executor(None, _clear_buffer)


# --- Original Code from Colab ---
def encode_audio_input(data: bytes, config: AudioConfig) -> dict:
    """Build JSPB message with user input audio bytes."""
    return {
        'realtimeInput': {
            'mediaChunks': [{
                'mimeType': f'audio/pcm;rate={config.sample_rate}',
                'data': base64.b64encode(data).decode('UTF-8'),
            }],
        },
    }

def encode_text_input(text: str) -> dict:
    """Builds JSPB message with user input text."""
    return {
        'clientContent': {
            'turns': [{
                'role': 'USER',
                'parts': [{'text': text}],
            }],
            'turnComplete': True,
        },
    }

def decode_audio_output(input: dict) -> bytes:
    """Returns byte string with model output audio."""
    result = []
    content_input = input.get('serverContent', {})
    content = content_input.get('modelTurn', {})
    for part in content.get('parts', []):
        data = part.get('inlineData', {}).get('data', '')
        if data:
            result.append(base64.b64decode(data))
    return b''.join(result)


async def main():
    """Main function to run the client."""
    async with contextlib.AsyncExitStack() as es:
        tg = await es.enter_async_context(asyncio.TaskGroup())
        audio_config = AudioConfig()
        audio = await es.enter_async_context(RunningLiveAudio(audio_config))
        
        uri = f'wss://{HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'
        conn = await es.enter_async_context(connect(uri))
        print('<connected>')

        initial_request = {
            'setup': {
                'model': MODEL,
            },
        }
        await conn.send(json.dumps(initial_request))

        if text := INITIAL_REQUEST_TEXT:
            print(f"Sending initial text: {text}")
            await conn.send(json.dumps(encode_text_input(text)))

        async def send_audio():
            while True:
                data = await audio.read()
                await conn.send(json.dumps(encode_audio_input(data, audio.config)))

        tg.create_task(send_audio())
        enqueued_audio = []
        
        print("Ready to talk...")
        async for msg in conn:
            msg = json.loads(msg)
            if to_play := decode_audio_output(msg):
                enqueued_audio.append(to_play)
                await audio.enqueue(to_play)  # enqueue TTS
            elif 'interrupted' in msg.get('serverContent', {}):
                print('<interrupted by the user>')
                await audio.clear_queue()  # stop TTS
            elif 'turnComplete' in msg.get('serverContent', {}):
                if enqueued_audio:  # save it for later playback
                    final_audio_data = b''.join(enqueued_audio)
                    filename = f"output_{int(time.time())}.wav"
                    print(f"Saving turn audio to {filename}")
                    sf.write(
                        filename,
                        np.frombuffer(final_audio_data, dtype=np.int16),
                        audio.config.sample_rate
                    )
                enqueued_audio = []
                print('<end of turn>')
            else:
                if msg != {'serverContent': {}}:
                    print(f'unhandled message: {msg}')


# --- main ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    except asyncio.CancelledError:
        print("Client stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
