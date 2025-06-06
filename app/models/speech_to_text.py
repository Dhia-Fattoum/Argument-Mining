import os
import gc
import json
import re
import torch
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from datetime import datetime
from typing import List, Dict, Optional
from pyannote.audio import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torchaudio
import whisper
import difflib
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

CONFIG = {
    "WHISPER_MODEL": "large-v3",
    "DIARIZATION_MODEL": "pyannote/speaker-diarization-3.1",
    "TEXT_ENHANCEMENT_MODEL": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "DEEPL_API_KEY": os.getenv("DEEPL_API_KEY", ""),
    "SAMPLE_RATE": 16000,
    "NOISE_REDUCTION_LEVEL": 0.2,
    "MIN_SPEAKER_DURATION": 0.5,
    "MIN_AUDIO_LENGTH": 1.0,
    "MAX_SPEAKERS": 5,
    "ENHANCE_TEMPERATURE": 0.7,
    "ENHANCE_TOP_P": 0.9,
    "ENHANCE_MAX_NEW_TOKENS": 512,
    "TARGET_LANGUAGE": "EN-US",
    "USE_GPU": torch.cuda.is_available(),
    "MAX_GPU_MEMORY": 0.9,
    "OUTPUT_DIR": "outputs",
    "ENABLE_TRANSLATION": True,
}

class AdvancedSpeechToText:
    def __init__(self):
        self.device = torch.device("cuda" if CONFIG["USE_GPU"] else "cpu")
        logging.info(f"Using device: {self.device}")
        self.models = {}
        self.translator = None
        self._validate_tokens()

    def _validate_tokens(self):
        if not CONFIG["HF_TOKEN"]:
            logging.error("Hugging Face token (HF_TOKEN) is required but not found.")
            raise ValueError("Hugging Face token is required.")
        
        if CONFIG["ENABLE_TRANSLATION"]:
            if not CONFIG["DEEPL_API_KEY"]:
                logging.warning("DeepL API key (DEEPL_API_KEY) not found. Translation will be skipped if attempted.")
            else:
                logging.info("DeepL API key found. Translation enabled.")
        else:
            logging.info("Translation is disabled via configuration (ENABLE_TRANSLATION=False).")

    def _setup_directories(self, base_output_dir: str):
        os.makedirs(base_output_dir, exist_ok=True)
        speakers_dir = os.path.join(base_output_dir, "speakers")
        transcripts_dir = os.path.join(base_output_dir, "transcripts")
        os.makedirs(speakers_dir, exist_ok=True)
        os.makedirs(transcripts_dir, exist_ok=True)
        logging.info(f"Output subdirectories ensured at: {speakers_dir} and {transcripts_dir}")

    def _validate_audio(self, audio_path: str) -> bool:
        try:
            info = sf.info(audio_path)
        except Exception as e:
            logging.error(f"Could not read audio info for {audio_path}: {e}")
            raise ValueError(f"Could not read audio info for {audio_path}: {e}")
            
        if info.duration < CONFIG["MIN_AUDIO_LENGTH"]:
            logging.error(f"Audio file {audio_path} is too short: {info.duration:.2f}s. Minimum: {CONFIG['MIN_AUDIO_LENGTH']}s.")
            raise ValueError(f"Audio too short: {info.duration:.2f}s. Minimum: {CONFIG['MIN_AUDIO_LENGTH']}s.")
        logging.info(f"Audio {audio_path} validated: duration {info.duration:.2f}s.")
        return True

    def _load_audio(self, audio_path: str) -> np.ndarray:
        logging.info(f"Loading audio from: {audio_path}")
        audio, orig_sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            logging.info("Audio converted to mono.")
        if orig_sr != CONFIG["SAMPLE_RATE"]:
            logging.info(f"Resampling audio from {orig_sr}Hz to {CONFIG['SAMPLE_RATE']}Hz.")
            resampler = torchaudio.transforms.Resample(orig_sr, CONFIG["SAMPLE_RATE"])
            audio_tensor = torch.from_numpy(audio).float()
            audio = resampler(audio_tensor).numpy()
        return audio

    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        self._validate_audio(audio_path)
        audio = self._load_audio(audio_path)
        logging.info("Applying noise reduction...")
        reduced_noise_audio = nr.reduce_noise(
            y=audio, sr=CONFIG["SAMPLE_RATE"], stationary=True, prop_decrease=CONFIG["NOISE_REDUCTION_LEVEL"]
        )
        logging.info("Noise reduction applied.")
        return reduced_noise_audio

 
    def _run_diarization(self, audio_path: str):
        if "diarization" not in self.models:
            logging.info(f"Loading diarization model: {CONFIG['DIARIZATION_MODEL']}")
            self.models["diarization"] = Pipeline.from_pretrained(
                CONFIG["DIARIZATION_MODEL"], use_auth_token=CONFIG["HF_TOKEN"]
            ).to(self.device)
            logging.info("Diarization model loaded.")
        
        logging.info(f"Starting speaker diarization for {audio_path}...")
        diarization_result = self.models["diarization"](
            audio_path, min_speakers=1, max_speakers=CONFIG["MAX_SPEAKERS"]
        )
        logging.info("Speaker diarization complete.")
        return diarization_result

    def _process_diarization(self, diarization) -> List[dict]:
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = float(turn.end - turn.start)
            if duration >= CONFIG["MIN_SPEAKER_DURATION"]:
                segments.append({
                    "start": float(turn.start), "end": float(turn.end),
                    "speaker": speaker, "duration": duration
                })
        logging.info(f"Processed {len(segments)} speaker segments after filtering.")
        return sorted(segments, key=lambda x: x["start"])

    def _group_speakers(self, segments: List[dict]) -> Dict[str, List[dict]]:
        grouped = {}
        for seg in segments:
            grouped.setdefault(seg["speaker"], []).append(seg)
        logging.info(f"Grouped segments for {len(grouped)} unique speakers.")
        return grouped

    def _create_speaker_audio(self, original_path: str, segments: List[dict], speaker: str, current_output_dir: str) -> str:
        logging.info(f"Creating combined audio file for speaker: {speaker}")
        try:
            audio_segment_full = AudioSegment.from_file(original_path)
        except Exception as e:
            logging.error(f"Could not load audio file {original_path} with pydub: {e}")
            raise
            
        speaker_audio = AudioSegment.empty()
        for seg in segments:
            speaker_audio += audio_segment_full[int(seg["start"] * 1000):int(seg["end"] * 1000)]
        
        out_path = os.path.join(current_output_dir, "speakers", f"{speaker.replace(' ', '_')}_audio.wav")
        speaker_audio.export(out_path, format="wav")
        logging.info(f"Exported audio for speaker {speaker} to: {out_path}")
        return out_path

    def _clean_transcript(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        return text[0].upper() + text[1:] if text else text

    def _transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> dict:
        if "whisper" not in self.models:
            logging.info(f"Loading Whisper model: {CONFIG['WHISPER_MODEL']}")
            self.models["whisper"] = whisper.load_model(CONFIG["WHISPER_MODEL"], device=self.device)
            logging.info("Whisper model loaded.")
        
        logging.info(f"Transcribing: {audio_path} (Lang: {'auto' if language is None else language})")
        result = self.models["whisper"].transcribe(audio_path, language=language, fp16=CONFIG["USE_GPU"])
        raw_text = result["text"]
        cleaned_text = self._clean_transcript(raw_text)
        logging.info(f"Transcription complete for {audio_path}.")
        return {"raw_text": raw_text, "cleaned_text": cleaned_text}

    def _load_text_enhancer(self):
        if "text_enhancer" not in self.models:
            logging.info(f"Loading text enhancement model: {CONFIG['TEXT_ENHANCEMENT_MODEL']}")
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["TEXT_ENHANCEMENT_MODEL"], token=CONFIG["HF_TOKEN"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["TEXT_ENHANCEMENT_MODEL"], device_map="auto",
                torch_dtype=torch.float16 if CONFIG["USE_GPU"] else torch.float32,
                load_in_4bit=True if CONFIG["USE_GPU"] else False, token=CONFIG["HF_TOKEN"]
            )
            self.models["text_enhancer"] = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
            )
            logging.info("Text enhancement model loaded.")

    def _strip_notes(self, text: str) -> str:
        return re.split(r'\bNote\b:|\bAdditional context\b:|\bExplanation\b:|\bAs requested\b:', text, flags=re.IGNORECASE)[0].strip()

    def _enhance_text(self, text: str) -> str:
        if not text.strip():
            logging.info("Skipping text enhancement for empty input.")
            return ""
        self._load_text_enhancer()

        prompt = f"""Your task is to meticulously review and enhance the following audio transcription. The goal is to produce a polished, highly readable text that is approximately 90% representative of the speaker's original intent, even if the initial transcription is imperfect.
Focus on these aspects:
1.  Correction: Correct any misrecognized words, names, or technical terms based on likely context.
2.  Completion: If words, letters, or parts of sentences seem missing or abruptly cut off, complete them in a way that aligns with the surrounding dialogue and the speaker's probable meaning.
3.  Fluency: Improve sentence structure and grammar for natural readability, resolving stutters or false starts if the core message can be preserved.
4.  Fidelity: Ensure the enhanced text remains true to the speaker's original message and tone. Do not add new information or opinions.

The output must ONLY be the final, enhanced transcription. Do not include any of your own explanations, notes, comments, apologies, or any prefix like "Enhanced Output:" or "Enhanced Transcription:".

Original Transcription:
\"\"\"
{text}
\"\"\"

Enhanced Transcription:"""

        logging.info(f"Enhancing text (first 50 chars): '{text[:50]}...'")
        output_list = self.models["text_enhancer"](
            prompt, temperature=CONFIG["ENHANCE_TEMPERATURE"], top_p=CONFIG["ENHANCE_TOP_P"],
            max_new_tokens=CONFIG["ENHANCE_MAX_NEW_TOKENS"], do_sample=True,
            pad_token_id=self.models["text_enhancer"].tokenizer.eos_token_id, num_return_sequences=1
        )
        generated_text = output_list[0]['generated_text']
        
        potential_enhanced_text = ""
        parts = generated_text.split("Enhanced Transcription:")

        if len(parts) > 1:
            potential_enhanced_text = parts[-1].strip()
            logging.info("Successfully extracted text using 'Enhanced Transcription:' marker.")
        elif generated_text.startswith(prompt):
            potential_enhanced_text = generated_text[len(prompt):].strip()
            logging.warning("Marker 'Enhanced Transcription:' not found. Extracted text by removing prompt. Review output carefully.")
        else:
            potential_enhanced_text = generated_text.strip()
            logging.warning(f"Marker 'Enhanced Transcription:' not found and prompt not at start. Using full generated text. Review output carefully. Potential output: '{potential_enhanced_text[:100]}...'")

        refusal_keywords = [
            "guidelines for this task", "not in the original text", "cannot fulfill", 
            "unable to process", "as an ai language model", "i am unable to",
            "i cannot provide"
        ]
        is_refusal = any(keyword.lower() in potential_enhanced_text.lower() for keyword in refusal_keywords)
        
        is_too_short_compared_to_input = len(text) > 50 and len(potential_enhanced_text) < (len(text) * 0.5)
        is_likely_fragment = len(potential_enhanced_text.split()) < 3 and not potential_enhanced_text.strip().endswith(('.', '!', '?'))

        if is_refusal or (is_too_short_compared_to_input and is_likely_fragment and not (len(text) < 20)):
            logging.error(f"Text enhancement likely failed or produced an off-topic/refusal response. LLM Output: '{potential_enhanced_text}'. Falling back to original cleaned text input to this function.")
            return text 

        enhanced_cleaned = self._clean_transcript(potential_enhanced_text)
        enhanced_final = self._strip_notes(enhanced_cleaned)
        
        if not enhanced_final.strip() and text.strip():
            logging.warning(f"Enhancement resulted in empty string after cleaning, while input was not empty. Original input: '{text}'. LLM output: '{potential_enhanced_text}'. Falling back to original cleaned text.")
            return text

        logging.info(f"Text enhancement complete. Result (first 50 chars): '{enhanced_final[:50]}...'")
        return enhanced_final

    def _calculate_text_similarity(self, reference: str, hypothesis: str) -> float:
        matcher = difflib.SequenceMatcher(None, reference.lower().split(), hypothesis.lower().split())
        similarity = round(matcher.ratio(), 4)
        logging.info(f"Calculated text similarity: {similarity}")
        return similarity

    def _translate_text(self, text: str) -> str:
        if not CONFIG["ENABLE_TRANSLATION"]:
            return "Translation disabled by configuration."
        if not CONFIG["DEEPL_API_KEY"]:
            return "Translation not available (API key missing)."
        if not text.strip(): return ""
        try:
            import deepl
            if not self.translator:
                logging.info("Initializing DeepL translator...")
                self.translator = deepl.Translator(CONFIG["DEEPL_API_KEY"])
            logging.info(f"Translating to {CONFIG['TARGET_LANGUAGE']} (first 50 chars): '{text[:50]}...'")
            result = self.translator.translate_text(text, target_lang=CONFIG["TARGET_LANGUAGE"])
            translated_text = self._strip_notes(self._clean_transcript(result.text))
            logging.info("Translation complete.")
            return translated_text
        except ImportError:
            logging.error("DeepL library not installed. pip install deepl")
            return "Translation error: DeepL library not installed."
        except Exception as e:
            logging.error(f"DeepL Translation error: {str(e)}")
            return f"Translation error: {str(e)}"

    def preload_models(self):
        logging.info("Starting model pre-loading routine...")
        if "diarization" not in self.models and CONFIG["HF_TOKEN"]:
            logging.info(f"Pre-loading diarization model: {CONFIG['DIARIZATION_MODEL']}")
            self.models["diarization"] = Pipeline.from_pretrained(
                CONFIG["DIARIZATION_MODEL"], use_auth_token=CONFIG["HF_TOKEN"]
            ).to(self.device)
            logging.info("Diarization model pre-loaded.")
        if "whisper" not in self.models:
            logging.info(f"Pre-loading Whisper model: {CONFIG['WHISPER_MODEL']}")
            self.models["whisper"] = whisper.load_model(CONFIG["WHISPER_MODEL"], device=self.device)
            logging.info("Whisper model pre-loaded.")
        self._load_text_enhancer()
        if CONFIG["ENABLE_TRANSLATION"] and CONFIG["DEEPL_API_KEY"] and not self.translator:
            try:
                import deepl
                logging.info("Pre-loading DeepL translator...")
                self.translator = deepl.Translator(CONFIG["DEEPL_API_KEY"])
                logging.info("DeepL translator pre-loaded.")
            except ImportError: logging.error("DeepL library not installed. Cannot pre-load translator.")
            except Exception as e: logging.error(f"Error pre-loading DeepL translator: {e}")
        logging.info("Model pre-loading routine complete.")

    def process_audio(self, audio_path: str, language: Optional[str] = None, output_dir_override: Optional[str] = None) -> List[dict]:
        current_output_dir = output_dir_override if output_dir_override else CONFIG["OUTPUT_DIR"]
        self._setup_directories(current_output_dir)

        logging.info(f"Starting full audio processing for: {audio_path}. Output to: {current_output_dir}")
        if CONFIG["USE_GPU"]: torch.cuda.empty_cache()

        try:
            _ = self._preprocess_audio(audio_path)
            diarization = self._run_diarization(audio_path)
            segments = self._process_diarization(diarization)
            if not segments:
                logging.warning(f"No speaker segments found for {audio_path}.")
                return [{"status": "failed", "error": "No speaker segments found.", "audio_file": audio_path}]

            speakers_grouped_segments = self._group_speakers(segments)
            results = []

            for speaker_id, speaker_segs in speakers_grouped_segments.items():
                logging.info(f"Processing segments for speaker: {speaker_id}")
                speaker_audio_path = self._create_speaker_audio(audio_path, speaker_segs, speaker_id, current_output_dir)
                transcription_data = self._transcribe_audio(speaker_audio_path, language)
                enhanced_text = self._enhance_text(transcription_data["cleaned_text"])
                translated_text = self._translate_text(enhanced_text)
                similarity_score = self._calculate_text_similarity(transcription_data["raw_text"], enhanced_text)

                result_entry = {
                    "speaker_id": speaker_id,
                    "audio_file_segments": speaker_audio_path,
                    "total_duration_speaker": sum(s["duration"] for s in speaker_segs),
                    "segment_count_speaker": len(speaker_segs),
                    "segments_timestamps": speaker_segs,
                    "transcripts": {
                        "raw": transcription_data["raw_text"], "cleaned": transcription_data["cleaned_text"],
                        "enhanced": enhanced_text, "enhancement_similarity_score": similarity_score,
                        "translation": translated_text,
                        "target_language": CONFIG["TARGET_LANGUAGE"] if CONFIG["ENABLE_TRANSLATION"] else "N/A"
                    }, "processing_timestamp": datetime.now().isoformat()
                }
                speaker_json_path = os.path.join(current_output_dir, "transcripts", f"{speaker_id.replace(' ', '_')}_transcript.json")
                with open(speaker_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_entry, f, indent=2, ensure_ascii=False)
                logging.info(f"Saved transcript for speaker {speaker_id} to {speaker_json_path}")
                results.append(result_entry)

            combined_fn = f"combined_results_{os.path.basename(audio_path).rsplit('.', 1)[0]}.json"
            combined_results_path = os.path.join(current_output_dir, combined_fn)
            combined_data = {
                "original_audio_file": audio_path, "detected_speakers": [r["speaker_id"] for r in results],
                "all_speaker_results": results, "overall_status": "complete",
                "overall_processing_timestamp": datetime.now().isoformat()
            }
            with open(combined_results_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved combined results to {combined_results_path}")
            return results

        except Exception as e:
            logging.error(f"Critical error for {audio_path}: {str(e)}", exc_info=True)
            error_fn = f"error_{os.path.basename(audio_path).rsplit('.', 1)[0]}.json"
            error_path = os.path.join(current_output_dir, error_fn)
            try:
                with open(error_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "original_audio_file": audio_path, "overall_status": "failed",
                        "error_message": str(e), "error_timestamp": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
                logging.info(f"Error details saved to {error_path}")
            except Exception as ex_save: logging.error(f"Could not save error JSON: {ex_save}")
            return [{"status": "failed", "error": str(e), "audio_file": audio_path}]
        finally:
            logging.info("Clearing models and attempting to free GPU cache.")
            self.models.clear()
            if self.translator: del self.translator; self.translator = None
            if CONFIG["USE_GPU"]: torch.cuda.empty_cache()
            gc.collect()
            logging.info("Cleanup complete.")

if __name__ == "__main__":
    logging.info("AdvancedSpeechToText script started (example usage block).")
    dummy_audio_path = "dummy_test_audio.wav"
    custom_output_path = "custom_run_outputs"

    if not os.path.exists(dummy_audio_path):
        try:
            sr = CONFIG["SAMPLE_RATE"]; dur = 3; freq = 440
            t = np.linspace(0, dur, int(sr*dur), False)
            audio_data = np.int16( (0.5*np.sin(2*np.pi*freq*t) + 0.02*np.random.randn(len(t))) * 32767)
            sf.write(dummy_audio_path, audio_data, sr)
            logging.info(f"Created dummy audio file: {dummy_audio_path}")
        except Exception as e:
            logging.error(f"Could not create dummy audio: {e}. Provide a valid audio file.")

    if not CONFIG["HF_TOKEN"]:
        logging.error("HF_TOKEN not set. Example will likely fail when loading models.")
    else:
        try:
            pipeline_instance = AdvancedSpeechToText()
            
            if os.path.exists(dummy_audio_path):
                logging.info(f"--- Processing with default output directory ({CONFIG['OUTPUT_DIR']}) ---")
                results_default = pipeline_instance.process_audio(dummy_audio_path)

                logging.info(f"--- Processing with custom output directory ({custom_output_path}) ---")
                results_custom = pipeline_instance.process_audio(dummy_audio_path, output_dir_override=custom_output_path)
            else:
                logging.warning(f"Audio file '{dummy_audio_path}' not found. Skipping example processing.")
        except ValueError as ve: logging.error(f"Initialization or validation error: {ve}")
        except Exception as e: logging.error(f"Unexpected error: {e}", exc_info=True)