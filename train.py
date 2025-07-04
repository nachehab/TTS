import os

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    output_path = os.path.dirname(os.path.abspath(__file__))

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=os.path.join(output_path, "WAV")
    )

    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=0,  # safer default for Windows; you can raise this later
        num_eval_loader_workers=0,
        run_eval=True,
        test_delay_epochs=-1,
        eval_split_size=0.1,
        epochs=250,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )

    trainer.fit()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for multiprocessing on Windows
    main()
