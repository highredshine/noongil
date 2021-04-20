package edu.brown.cs.student.noongil.noongil;

import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import spark.Request;
import spark.Response;
import spark.Route;
import spark.Spark;
import org.json.JSONObject;

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
      JSONObject obj = new JSONObject(request.body());
      String base64img = obj.getString("input").replace("data:image/png;base64,","");
      // TODO: decode the base64 encoded img

      // TODO: send the img data to python / save it as a file for python code to read

      // TODO: retrieve the result output (text read from the image)

      // TODO: send it as response
      return GSON.toJson(ImmutableMap.of("output", "it's my face"));
    }
  }
}
