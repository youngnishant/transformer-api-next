import * as ort from "onnxruntime-node";

export class PretrainedModel {
  async loadSession(modelBuffer) {
    // const response = await fetch(modelSource, { cache: "force-cache" });
    // const modelBuffer = await response.arrayBuffer();
    const session = await ort.InferenceSession.create(modelBuffer);
    return session;
  }
}
