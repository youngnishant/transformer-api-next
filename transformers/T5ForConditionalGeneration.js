import * as ort from "onnxruntime-node";

import { AutoModelForSeq2SeqLM } from "./AutoModelForSeq2SeqLM.js";
import { Seq2SeqLMOutput } from "./Seq2SeqLMOutput.js";

export class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
  constructor(modelId, modelsPath, progressAsyncCallback = undefined) {
    super(modelId, modelsPath, progressAsyncCallback);
  }

  async loadSessions() {
    const modelIdParts = this.modelId.split("/");
    const modelName = modelIdParts[modelIdParts.length - 1];
    const suffix = "-quantized";
    const encoderUrl = `${this.modelsPath}/${modelName}-encoder${suffix}.onnx`;
    const initDecoderUrl = `${this.modelsPath}/${modelName}-init-decoder${suffix}.onnx`;
    const decoderUrl = `${this.modelsPath}/${modelName}-decoder${suffix}.onnx`;

    const progressMax = 4;
    let progress = 0;
    const incrementProgress = async () => {
      progress++;
      const p = progress / progressMax;
      if (this.progressAsyncCallback) {
        await this.progressAsyncCallback(p);
      }
    };
    await incrementProgress();
    const encoderSessionPromise = this.loadSession(encoderUrl);
    const initDecoderSessionPromise = this.loadSession(initDecoderUrl);
    const decoderSessionPromise = this.loadSession(decoderUrl);
    const encoderSession = await encoderSessionPromise;
    await incrementProgress();
    const initDecoderSession = await initDecoderSessionPromise;
    await incrementProgress();
    const decoderSession = await decoderSessionPromise;
    await incrementProgress();
    return [encoderSession, initDecoderSession, decoderSession];
  }

  async forward(inputIds, decoderInputIds, encoderOutputs, pastKeyValues) {
    // console.log(new ort.Tensor());
    const inputIdsTensor = new ort.Tensor(
      "int64",
      new BigInt64Array(inputIds.map((x) => BigInt(x))),
      [1, inputIds.length]
    );
    const encoderAttentionMaskTensor = new ort.Tensor(
      "int64",
      new BigInt64Array(inputIds.length).fill(BigInt(1)),
      [1, inputIds.length]
    );
    const [encoderSession, initDecoderSession, decoderSession] =
      await this.getSessions();
    if (encoderOutputs === null) {
      const encoderFeeds = {
        input_ids: inputIdsTensor,
        attention_mask: encoderAttentionMaskTensor,
      };
      const encoderResults = await encoderSession.run(encoderFeeds);
      const encoderHiddenStates = encoderResults.hidden_states;
      encoderOutputs = encoderHiddenStates;
    }

    const decoderInputIdsTensor = new ort.Tensor(
      "int64",
      new BigInt64Array(decoderInputIds.map((x) => BigInt(x))),
      [1, decoderInputIds.length]
    );
    const decoderFeeds = {
      input_ids: decoderInputIdsTensor,
      encoder_attention_mask: encoderAttentionMaskTensor,
      encoder_hidden_states: encoderOutputs,
    };
    let logits = null;

    if (pastKeyValues === null) {
      const initDecoderResults = await initDecoderSession.run(decoderFeeds);
      logits = initDecoderResults.logits;
      pastKeyValues = this.getPastKeyValues(
        initDecoderSession.outputNames.slice(1),
        initDecoderResults
      );
    } else {
      for (const p of pastKeyValues) {
        decoderFeeds[p.name] = p.data;
      }
      const decoderResults = await decoderSession.run(decoderFeeds);
      logits = decoderResults.logits;
      pastKeyValues = this.getPastKeyValues(
        decoderSession.outputNames.slice(1),
        decoderResults
      );
    }
    return new Seq2SeqLMOutput(logits, pastKeyValues, encoderOutputs);
  }

  getPastKeyValues(pkvNames, decoderResults) {
    const pkvs = [];
    for (const i in pkvNames) {
      const k = pkvNames[i];
      const v = decoderResults[k];
      pkvs.push({ name: `pkv_${i}`, data: v });
    }
    return pkvs;
  }
}
