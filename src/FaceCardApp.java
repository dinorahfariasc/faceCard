import org.opencv.core.*;
import org.opencv.face.*;
import org.opencv.videoio.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.json.*;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;


import java.io.*;
import java.util.*;

public class FaceCardApp {
    private static final String JSON_FILE_PATH = "faces.json";
    private static VideoCapture capture;
    private static FaceRecognizer faceRecognizer;
    private static List<Mat> facesList = new ArrayList<>();
    private static List<Integer> labelsList = new ArrayList<>();

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println("Erro ao acessar a webcam!");
            return;
        }

        // Inicializa o reconhecedor facial
        faceRecognizer = LBPHFaceRecognizer.create();

        // Carregar faces do arquivo JSON
        loadFacesFromJson();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("Escolha uma opção:");
            System.out.println("1. Cadastrar nova face");
            System.out.println("2. Reconhecer face");
            System.out.println("3. Sair");

            int option = scanner.nextInt();
            if (option == 1) {
                System.out.println("Cadastrando nova face...");
                captureFace();
            } else if (option == 2) {
                System.out.println("Reconhecendo face...");
                recognizeFace();
            } else if (option == 3) {
                saveFacesToJson();
                break;
            }
        }
        capture.release();
        HighGui.destroyAllWindows();
    }

    // Captura uma face e salva
    private static void captureFace() {
        Mat frame = new Mat();
        capture.read(frame);

        // Converte para escala de cinza
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

        // Detecta faces
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(grayFrame, faces);

        for (Rect rect : faces.toArray()) {
            // Extraímos a face detectada
            Mat face = new Mat(frame, rect);
            facesList.add(face);
            labelsList.add(facesList.size() - 1);  // Associe um identificador à face

            // Treine o reconhecedor
            faceRecognizer.update(facesList, new MatOfInt(labelsList.stream().mapToInt(i -> i).toArray()));

            // Exibe a face capturada
            Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            HighGui.imshow("Face Capturada", frame);
            HighGui.waitKey(1000); // Aguarda 1 segundo para capturar a face
        }
    }

    // Reconhece a face usando o modelo treinado
    private static void recognizeFace() {
        Mat frame = new Mat();
        capture.read(frame);

        // Converte para escala de cinza
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

        // Detecta faces
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(grayFrame, faces);

        for (Rect rect : faces.toArray()) {
            // Extraímos a face detectada
            Mat face = new Mat(frame, rect);

            // Reconhece a face
            int predictedLabel = faceRecognizer.predict_label(face);
            String label = "Desconhecido";

            if (predictedLabel != -1) {
                label = "Face reconhecida com label: " + predictedLabel;
            }

            // Exibe o resultado
            Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            Imgproc.putText(frame, label, new Point(rect.x, rect.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);

            HighGui.imshow("Reconhecimento", frame);
        }
    }

    // Salva as faces e labels em um arquivo JSON
    private static void saveFacesToJson() {
        JSONObject jsonObject = new JSONObject();
        JSONArray facesArray = new JSONArray();

        for (int i = 0; i < facesList.size(); i++) {
            JSONObject faceObj = new JSONObject();
            faceObj.put("label", labelsList.get(i));
            faceObj.put("face", encodeMatToBase64(facesList.get(i)));
            facesArray.put(faceObj);
        }

        jsonObject.put("faces", facesArray);

        try (FileWriter file = new FileWriter(JSON_FILE_PATH)) {
            file.write(jsonObject.toString());
            System.out.println("Faces salvas no arquivo JSON.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Carrega as faces do arquivo JSON
    private static void loadFacesFromJson() {
        try (FileReader reader = new FileReader(JSON_FILE_PATH)) {
            StringBuilder jsonContent = new StringBuilder();
            int c;
            while ((c = reader.read()) != -1) {
                jsonContent.append((char) c);
            }

            JSONObject jsonObject = new JSONObject(jsonContent.toString());
            JSONArray facesArray = jsonObject.getJSONArray("faces");

            for (int i = 0; i < facesArray.length(); i++) {
                JSONObject faceObj = facesArray.getJSONObject(i);
                int label = faceObj.getInt("label");
                String base64Face = faceObj.getString("face");

                // Decodificar a face e adicionar à lista
                Mat face = decodeBase64ToMat(base64Face);
                facesList.add(face);
                labelsList.add(label);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Codifica uma Mat em Base64 (para salvar em JSON)
    private static String encodeMatToBase64(Mat mat) {
        // Use uma biblioteca para converter a Mat em Base64
        // Exemplo com OpenCV para converter em um vetor de bytes
        return Base64.getEncoder().encodeToString(mat.dataAddr());
    }

    // Decodifica uma Mat de Base64 (para carregar do JSON)
    private static Mat decodeBase64ToMat(String base64) {
        // Usar uma função para converter de Base64 para Mat
        return new Mat();
    }
}
