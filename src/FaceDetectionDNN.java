import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class FaceDetectionDNN {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Carregar o modelo e o arquivo de configuração do DNN
        String modelFile = "path/to/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        String configFile = "path/to/deploy.prototxt";
        Net net = Dnn.readNetFromCaffe(configFile, modelFile);

        // Iniciar captura de vídeo (webcam)
        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println("Erro ao abrir a webcam!");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            if (capture.read(frame)) {
                // Converter o frame para um blob, a entrada necessária para a rede
                Mat blob = Dnn.blobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0), false, false);
                net.setInput(blob);

                // Detectar as faces
                Mat detections = net.forward();

                // O modelo retorna uma matriz 4D (batch_size, 1, N, 7) onde:
                // batch_size: número de imagens no lote (sempre 1 neste caso),
                // 1: número de canais (sempre 1),
                // N: número de detecções de faces,
                // 7: valores para cada detecção (confiança, x1, y1, x2, y2)

                for (int i = 0; i < detections.size(2); i++) {
                    // A matriz de detecção tem um formato (1, 1, N, 7), portanto, pegamos a linha [0, 0, i]
                    float[] detectionData = detections.get(0, 0, i);  // 7 elementos para cada detecção

                    if (detectionData != null && detectionData.length >= 7) {
                        float confidence = detectionData[2]; // Confiança da detecção
                        if (confidence > 0.5) {  // Defina um limite de confiança (50%)
                            // As coordenadas x1, y1, x2, y2 para a caixa delimitadora (bounding box)
                            float x1 = detectionData[3] * frame.cols();
                            float y1 = detectionData[4] * frame.rows();
                            float x2 = detectionData[5] * frame.cols();
                            float y2 = detectionData[6] * frame.rows();

                            // Garanta que as coordenadas estão dentro dos limites
                            x1 = Math.max(0, Math.min(x1, frame.cols() - 1));
                            y1 = Math.max(0, Math.min(y1, frame.rows() - 1));
                            x2 = Math.max(0, Math.min(x2, frame.cols() - 1));
                            y2 = Math.max(0, Math.min(y2, frame.rows() - 1));

                            // Desenhe a caixa delimitadora ao redor da face
                            Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2);
                        }
                    }
                }

                // Exibir o frame com as detecções
                HighGui.imshow("Face Detection", frame);
                if (HighGui.waitKey(30) >= 0) {
                    break; // Sai do loop ao pressionar qualquer tecla
                }
            }
        }
        capture.release();
        HighGui.destroyAllWindows();
    }
}
