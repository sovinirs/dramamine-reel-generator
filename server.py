from flask import Flask, request, send_file, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from moviepy import (
    AudioFileClip,
)
import script
from collections import namedtuple

TranscriptionWord = namedtuple("TranscriptionWord", ["start", "end", "word"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("output", exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "mp4", "m4a"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Process the audio file using your existing script
            language = request.form.get("language", "en")
            output_path = os.path.join("output", f"processed_{filename}.mp4")

            # Reference the existing script's main processing logic
            result = script.transcribe_audio_with_timestamps(filepath, language)

            if language != "en":
                script_type = "devanagari" if language == "hi" else "tamil script"
                wordsList = result.words
                text_for_transliteration = result.text
                romanized_output = script.transliterate_transcription(
                    text_for_transliteration, language, script_type
                )

                # Create new TranscriptionWord objects with romanized words
                romanized_words = []
                for i, word_obj in enumerate(wordsList):
                    if i < len(romanized_output):
                        romanized_words.append(
                            TranscriptionWord(
                                start=word_obj.start,
                                end=word_obj.end,
                                word=romanized_output[i],
                            )
                        )

                video_segments = romanized_words

            else:
                video_segments = result.words

            # Process segments and create video
            init_segments = script.initial_segmentation(video_segments)
            merged_segments = script.merge_segments(init_segments)
            refined_segments = script.refine_segments_by_word_count(merged_segments)
            final_output = script.convert_to_tuple_format(refined_segments)

            video_paths = [f for f in os.listdir("./videos") if f.endswith(".mp4")]
            video_paths = [os.path.join("./videos", f) for f in video_paths]

            audio_file = AudioFileClip(filepath)

            script.process_and_stitch_videos_with_segments(
                video_paths, final_output, output_path, audio_file
            )

            return jsonify({"success": True, "filename": f"Dramamine_{filename}.mp4"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join("output", filename), as_attachment=True)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
