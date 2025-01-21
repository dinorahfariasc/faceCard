import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

public class WebcamTest {
    public static void main(String[] args) {
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);

        VideoCapture camera = new VideoCapture(0); // 0 para a webcam padrão
        if (!camera.isOpened()) {
            System.out.println("Erro ao abrir a câmera!");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            if (camera.read(frame)) {
                System.out.println("Frame capturado: " + frame);
                // Aqui você pode adicionar código para exibir o frame ou processá-lo
            }
        }
    }
}
