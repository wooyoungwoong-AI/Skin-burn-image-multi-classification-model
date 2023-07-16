/*
 * Created by ishaanjav
 * github.com/ishaanjav
 */

package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;


public class MainActivity extends AppCompatActivity {

    Button emergencyButton, camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        emergencyButton = findViewById(R.id.emergencyButton);
        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        emergencyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("응급 처치");
                builder.setCancelable(true);

                builder.setPositiveButton("1도", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        String message = "너무 경미한 수준의 화상이라 병원에 갈 필요없다. 다만 바로 차가운 물에 담갔다간 물집이 안 나도 될 화상에 물집이 생겨버린다. 그러니까 미지근한 물에서 찬 물로, 차근차근 천천히 온도를 낮춰나가야 한다. 찬물을 쓰지 말라는 게 아니다. 비록 원인은 다르지만 동상의 치료법과 비슷하다.";
                        displayMessage(message);
                    }
                });

                builder.setNeutralButton("2도", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        String message = "여기까지는 병원 치료를 동반할 경우 기능적으로 완전한 치유가 가능하다고는 하지만, 아무리 미온수를 들이부어도 작열감이 사라지지 않기 때문에 1도 화상과 달리 최소 병원에 가서 진료를 받고 약은 처방받아야 하고, 경우에 따라서는 입원 치료를 받아야 할 수 있다.";
                        displayMessage(message);
                    }
                });

                builder.setNegativeButton("3도", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        String message = "한시라도 빨리 병원을 방문 해야 합니다.";
                        displayMessage(message);
                    }
                });

                AlertDialog dialog = builder.create();
                dialog.show();
            }
        });

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"1도", "2도", "3도"};

            String resultText = "Class: " + classes[maxPos] + "\n";

            result.setText(resultText);

            model.close();

            // Display the result as a toast message with a delay of 3 seconds
            Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, resultText, Toast.LENGTH_SHORT).show();
                }
            }, 3000);

        } catch (IOException e) {
            // Exception handling
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void displayMessage(String message) {
        result.setText(message);
    }
}
