import os
import random

from openai import OpenAI
from dotenv import load_dotenv
from collections import namedtuple
from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    AudioFileClip,
    vfx,
    afx,
)

load_dotenv()

TranscriptionWord = namedtuple("TranscriptionWord", ["start", "end", "word"])

# Replace with your OpenAI API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# A list of common determiners (or similar markers) that often indicate an incomplete phrase.
DETERMINERS = {
    "a",
    "an",
    "the",
    "any",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}


def transcribe_audio_with_timestamps(audio_file_path, language="en"):
    """
    Transcribes the audio file and returns a verbose JSON response
    including segment-level timestamps.

    Parameters:
        audio_file_path (str): Path to the audio file.
        language (str): Optional language parameter (e.g., "tamil", "english", "hindi").

    Returns:
        dict: JSON response from the Whisper API containing transcription and timestamps.
    """
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",  # Returns segments with timestamps
            language=language,
            timestamp_granularities=["word"],  # e.g., "tamil", "english", "hindi"
        )
    return transcript


def transliterate_transcription(input_str: str, language: str, script: str) -> list:
    messages = [
        {
            "role": "user",
            "content": (
                f"You are a helpful assistant that transliterates {language} text written in {script}. "
                f"Transliterate the following {language} text into romanized text (Latin script).\n\n"
                "1. If a word is not in the dictionary, choose the closest meaning based on context.\n"
                '2. Do not a break any word into multiple words (example: "जिसमें" should be "jisme" and not "jis me"). '
                "The number of words in response should be the same as the number of words in input.\n\n"
                "Provide only the romanized text, no additional text:\n\n"
                f'"""\n{input_str}\n"""'
            ),
        },
    ]

    response = client.chat.completions.create(
        model="o1-preview",
        messages=messages,
    )

    content = response.choices[0].message.content
    words = content.translate(str.maketrans("", "", ".,!?")).split()
    return words


###########################################
# STEP 1: Initial Segmentation by Pause
###########################################
def initial_segmentation(words, pause_threshold=0.3):
    """
    Splits the list of TranscriptionWord objects into segments whenever the gap
    between consecutive words exceeds the pause_threshold.
    """
    segments = []
    current_segment = []

    for i, word in enumerate(words):
        if i == 0:
            current_segment.append(word)
        else:
            gap = word.start - words[i - 1].end
            if gap > pause_threshold:
                segments.append(current_segment)
                current_segment = [word]
            else:
                current_segment.append(word)

    if current_segment:
        segments.append(current_segment)
    return segments


###########################################
# STEP 2: Merge Segments Based on Heuristics
###########################################
def should_merge(segment1, segment2):
    """
    Decide whether two segments should be merged based on a simple heuristic:
    if the last word of segment1 is in our list of determiners,
    it indicates that the phrase may be incomplete and should be merged.
    """
    if not segment1 or not segment2:
        return False

    last_word = segment1[-1].word.lower()
    return last_word in DETERMINERS


def merge_segments(segments):
    """
    Merge adjacent segments if the last word of the earlier segment is a determiner.
    """
    if not segments:
        return segments

    merged_segments = [segments[0]]
    for seg in segments[1:]:
        if should_merge(merged_segments[-1], seg):
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)
    return merged_segments


###########################################
# STEP 3: Enforce Word Count Constraints
###########################################
def split_long_segment(segment, max_words, min_words):
    """
    Splits a segment (a list of words) into chunks that are no longer than max_words.
    When possible, it tries to avoid ending a chunk on a determiner.
    Even if a chunk is as short as min_words, we keep it if it represents a sentence-complete phrase.
    """
    chunks = []
    i = 0
    while i < len(segment):
        # Tentatively take up to max_words
        end_index = min(i + max_words, len(segment))
        # If the chunk ends on a determiner and the chunk is longer than min_words,
        # try to shift the break point backward to get a "complete" ending.
        if (
            segment[end_index - 1].word.lower() in DETERMINERS
            and (end_index - i) > min_words
        ):
            j = end_index - 1
            while j > i and segment[j].word.lower() in DETERMINERS:
                j -= 1
            # If we found a break point that leaves at least min_words, use it.
            if j > i and (j - i + 1) >= min_words:
                end_index = j + 1
        chunks.append(segment[i:end_index])
        i = end_index
    return chunks


def split_segments_if_too_long(segments, max_words, min_words):
    """
    For each segment that has more than max_words, split it using split_long_segment.
    """
    new_segments = []
    for seg in segments:
        if len(seg) > max_words:
            new_segments.extend(split_long_segment(seg, max_words, min_words))
        else:
            new_segments.append(seg)
    return new_segments


def merge_small_segments(segments, min_words):
    """
    Merge adjacent segments **only if** the segment is too short
    (fewer than min_words) and appears incomplete (ends with a determiner).
    If a segment is short but ends with a non-determiner, we assume it's a complete sentence.
    """
    if not segments:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        # Check if current segment is short and incomplete
        if len(current) < min_words and current[-1].word.lower() in DETERMINERS:
            # Merge with next segment if available
            if i < len(segments) - 1:
                current = current + segments[i + 1]
                i += 2
            else:
                i += 1
            merged.append(current)
        else:
            merged.append(current)
            i += 1
    return merged


def refine_segments_by_word_count(segments, min_words=6, max_words=10):
    """
    First, enforce sentence breaks from earlier heuristics.
    Then, if a segment exceeds max_words, split it.
    Finally, merge segments that are too short **only if** they appear incomplete.
    The sentence heuristic (i.e. merging based on determiners) has precedence.
    """
    # Step 1: Split any segment that is too long.
    segments = split_segments_if_too_long(segments, max_words, min_words)
    # Step 2: Merge small segments only if incomplete (ends with a determiner).
    segments = merge_small_segments(segments, min_words)
    return segments


###########################################
# STEP 4: Convert to Desired Output Format
###########################################
def convert_to_tuple_format(segments):
    """
    Converts each segment (a list of TranscriptionWord objects) into a list of tuples,
    each tuple containing (word, start_timestamp).
    """
    final = []
    for seg in segments:
        final_seg = [(word.word, word.start) for word in seg]
        final.append(final_seg)
    return final


###########################################
# STEP 5: Helper Functions for Text Placement
###########################################
def get_vertical_positions(num_words, video_height, text_heights):
    """
    Calculate vertical positions for words in a segment, evenly distributed from top to bottom.
    Returns list of y positions for each word.
    """
    total_text_height = sum(text_heights)
    spacing = (video_height - total_text_height) / (num_words + 1)

    positions = []
    current_y = spacing  # Start after first spacing
    for height in text_heights:
        positions.append(current_y)
        current_y += height + spacing
    return positions


def get_random_x_position(text_width, video_width):
    """Get random x position ensuring text stays within video width"""
    # Add padding of 20 pixels on each side to prevent text from touching edges
    padding = 20
    return random.randint(padding, video_width - text_width - padding)


###########################################
# STEP 6: Process and Stitch Videos with Segments
###########################################
def process_and_stitch_videos_with_segments(
    video_paths, final_output, stitched_output_path, audio_file
):
    """
    Processes videos by overlaying text segments and stitches them together.

    Parameters:
      video_paths (list): List of input video file paths to randomly select from
      final_output (list): List of segments, where each segment is a list of tuples (word, start_timestamp)
      stitched_output_path (str): File path for the final stitched video
    """
    # Define text styling
    font_list = [f for f in os.listdir("./fonts") if f.endswith((".ttf", ".otf"))]
    font_list = [os.path.join("./fonts", f) for f in font_list]

    colors = ["#ebd668", "#fefcfa"]

    # Target resolution for all videos (portrait mode)
    target_resolution = (608, 1080)  # width x height

    processed_clips = []
    used_video_paths = set()
    available_video_paths = video_paths.copy()

    final_output_sampled = final_output

    # Process each segment
    for i, segment in enumerate(final_output_sampled):
        # If we've used all videos, reset the available videos
        if not available_video_paths:
            available_video_paths = [
                p for p in video_paths if p not in used_video_paths
            ]
            if not available_video_paths:  # If still empty, reset completely
                available_video_paths = video_paths.copy()
                used_video_paths.clear()

        # Randomly select an unused video
        video_path = random.choice(available_video_paths)
        available_video_paths.remove(video_path)
        used_video_paths.add(video_path)

        # Load and prepare video
        video = VideoFileClip(video_path)

        # Convert to portrait if needed
        # optional and can be removed
        if video.size[0] > video.size[1]:
            video = video.rotated(90)

        # Resize/crop to target resolution
        video = video.resized(target_resolution)

        # Get segment duration based on start of next segment or add 1 second pause for last segment
        if i < len(final_output_sampled) - 1:
            segment_duration = (
                final_output[i + 1][0][1] - segment[0][1]
            )  # Next segment's first word start - current segment's first word start
        else:
            segment_duration = (
                segment[-1][1] - segment[0][1] + 1.0
            )  # Add 1 second pause for last segment

        # Trim video to segment duration
        video = video.subclipped(0, segment_duration)

        # Add moody filter to the video contrasting text with colors (#ebd668 & #fefcfa)
        video = vfx.LumContrast(lum=-50, contrast=0.3).apply(video)
        video = vfx.GammaCorrection(1.2).apply(video)
        video = vfx.MultiplyColor(factor=0.85).apply(video)

        clips = [video]

        # First create all text clips to get their heights
        text_clips = []
        text_heights = []
        for word, _ in segment:
            font = random.choice(font_list)
            color = random.choice(colors)

            # Start with a base font size and adjust until we get desired height
            target_height_in_pt = random.randint(
                65, 70
            )  # Target height similar to Arial 40-50pt
            font_size = target_height_in_pt  # Initial guess

            # Create test clip to measure height
            test_clip = TextClip(
                text=word,
                font_size=font_size,
                font="./fonts/Arial.ttf",
                color=color,
                margin=(0, 20),
            )

            # Adjust font size based on actual height
            actual_height = test_clip.size[1]
            adjusted_font_size = int(font_size * (target_height_in_pt / actual_height))

            # Create final clip with adjusted size
            txt_clip = TextClip(
                text=word,
                font_size=adjusted_font_size,
                font=font,
                color=color,
                margin=(0, 20),
            )

            text_clips.append(txt_clip)
            text_heights.append(txt_clip.size[1])

        # Get vertical positions for all words
        y_positions = get_vertical_positions(
            len(segment), target_resolution[1], text_heights
        )

        # Add words with timestamps
        for idx, ((word, start_time), txt_clip, y_pos) in enumerate(
            zip(segment, text_clips, y_positions)
        ):
            # Get random x position
            x_pos = get_random_x_position(txt_clip.size[0], target_resolution[0])

            # Set position and timing
            txt_clip = (
                txt_clip.with_position((x_pos, y_pos))
                .with_start(start_time - segment[0][1])  # Relative to segment start
                .with_duration(
                    segment_duration - (start_time - segment[0][1])
                )  # Until end
            )
            clips.append(txt_clip)

        # Create composite for this segment
        composite_clip = CompositeVideoClip(clips, size=target_resolution)
        processed_clips.append(composite_clip)

    # Concatenate segments and combine audio tracks
    final_video = concatenate_videoclips(processed_clips, method="compose")
    music_file = (
        AudioFileClip("./audio/DramamineFM.mp3")
        .with_duration(final_video.duration)
        .with_effects([afx.MultiplyVolume(0.7)])  # Decrease volume by 30%
    )
    audio_file = audio_file.with_effects(
        [afx.MultiplyVolume(1.3)]
    )  # Increase volume by 30%
    final_video = final_video.with_audio(CompositeAudioClip([audio_file, music_file]))

    final_video.write_videofile(
        stitched_output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )
