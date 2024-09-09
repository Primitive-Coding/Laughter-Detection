from laughter_detection import LaughterDetection


if __name__ == "__main__":

    l = LaughterDetection(output_dir="./InEdit")
    print(l.get_transcription())
    # r = l.process_video("trump.mp4")

    # r.to_csv("./InEdit/transcript.csv")
