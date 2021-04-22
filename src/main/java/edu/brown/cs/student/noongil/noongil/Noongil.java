package edu.brown.cs.student.noongil.noongil;

import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import org.apache.commons.io.FileUtils;
import spark.Request;
import spark.Response;
import spark.Route;
import spark.Spark;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Base64;
import java.util.Scanner;

public class Noongil {

  private static final Gson GSON = new Gson();

  public Noongil() {
    Spark.post("/input", new InputHandler());
  }

  private static class InputHandler implements Route {

    /**
     * Invoked when a request is made on this route's corresponding path e.g. '/hello'
     *
     * @param request  The request object providing information about the HTTP request
     * @param response The response object providing functionality for modifying the response
     * @return The content to be set in the response
     * @throws Exception implementation can choose to throw exception
     */
    @Override
    public Object handle(Request request, Response response) throws Exception {
      // get request body
      JSONObject obj = new JSONObject(request.body());
      String base64img = obj.getString("input").replace("data:image/png;base64,","");
      // decode the base64 encoded string to png image
      System.out.println("detected");
      this.saveImage(base64img);
      // run python code
      String command = "python tr/src/main.py --predict tr/data/image.png";
      Runtime.getRuntime().exec(command);
      // TODO: retrieve the result output (text read from the image)
      File outputTxt = new File("tr/model/recognized.txt");
      Scanner myReader = new Scanner(outputTxt);
      String output = myReader.nextLine();
      // TODO: send it as response
      return GSON.toJson(ImmutableMap.of("output", output));
    }

    public void saveImage(String encodedString) throws IOException {
      // create output file
      File outputFile = new File("tr/data/image.png");

      // decode the string and write to file
      byte[] decodedBytes = Base64.getDecoder().decode(encodedString);
      FileUtils.writeByteArrayToFile(outputFile, decodedBytes);
    }


  }
}
