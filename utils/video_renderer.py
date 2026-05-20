import cv2



def create_captioned_video(
    input_video,
    output_video,
    caption
):

    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (20, height - 140),
            (width - 20, height - 20),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.6,
            frame,
            0.4,
            0
        )

        y = height - 100

        words = caption.split()

        lines = []

        current = ""

        for word in words:

            test = current + " " + word

            if len(test) < 55:
                current = test

            else:
                lines.append(current)
                current = word

        lines.append(current)

        for line in lines:

            cv2.putText(
                frame,
                line,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 35

        out.write(frame)

    cap.release()
    out.release()