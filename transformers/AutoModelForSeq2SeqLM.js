import { PretrainedModel } from "./PretrainedModel.js";

export class AutoModelForSeq2SeqLM extends PretrainedModel {
  modelId;
  modelsPath;
  progressAsyncCallback;
  sessions = [];

  constructor(modelId, modelsPath, progressAsyncCallback) {
    super();
    this.modelId = modelId;
    this.modelsPath = modelsPath;
    this.progressAsyncCallback = progressAsyncCallback;
  }

  async getSessions() {
    if (this.sessions.length > 0) {
      return this.sessions;
    }
    this.sessions = await this.loadSessions();
    return this.sessions;
  }

  // loadSessions()

  async generate(inputTokenIds, options, progressAsyncCallback = undefined) {
    const maxLength = options.maxLength || 100;
    const topK = options.topK || 0;
    const topP = options.topP || 0;
    const numBeams = options.numBeams || 0;
    // attention_mask=token['attention_mask'], num_beams=2
    const startOfDecoderTokenId = 0;
    const endOfDecoderTokenId = 1;
    let encoderOutputs = null;
    let pastKeyValues = null;
    const outputTokenIds = [startOfDecoderTokenId];
    let numOutputTokens = 1;
    let shouldContinue = true;
    const maxOutputTokens = numOutputTokens + maxLength;
    async function progress() {
      if (progressAsyncCallback) {
        shouldContinue = await progressAsyncCallback(
          outputTokenIds,
          inputTokenIds
        );
      }
    }
    let sampler = (x) => this.sampleLogitsGreedily(x);
    if (topK > 0) {
      sampler = (x) => this.sampleLogitsTopK(x, topK);
    }
    while (shouldContinue && numOutputTokens < maxOutputTokens) {
      const output = await this.forward(
        inputTokenIds,
        outputTokenIds,
        encoderOutputs,
        pastKeyValues
      );

      pastKeyValues = output.pastKeyValues;
      encoderOutputs = output.encoderOutputs;
      const newTokenId = sampler(output.logits);
      outputTokenIds.push(newTokenId);
      numOutputTokens++;
      await progress();
      if (newTokenId === endOfDecoderTokenId) {
        break;
      }
    }
    return outputTokenIds;
  }

  sampleLogitsGreedily(logits) {
    const shape = logits.dims;
    const [batchSize, seqLength, vocabSize] = shape;
    const n = batchSize * seqLength * vocabSize;
    const startIndex = n - vocabSize;
    let argmaxi = 0;
    let argmax = logits.data[startIndex + argmaxi];
    for (let i = 1; i < vocabSize; i++) {
      const l = logits.data[startIndex + i];
      if (l > argmax) {
        argmaxi = i;
        argmax = l;
      }
    }
    return argmaxi;
  }

  sampleLogitsTopK(logits, k) {
    const shape = logits.dims;
    const [batchSize, seqLength, vocabSize] = shape;
    const n = batchSize * seqLength * vocabSize;
    const startIndex = n - vocabSize;
    const logs = logits.data.slice(startIndex);
    k = Math.min(k, vocabSize);
    const logitAndId = Array.from(logs)
      .map((x, i) => [x, i])
      .sort((a, b) => b[0] - a[0]);
    const sMin = Math.exp(-100.0);
    let sumS = 0.0;
    for (let i = 0; i < logitAndId.length; i++) {
      const s = i < k ? Math.exp(logitAndId[i][0]) : sMin;
      sumS += s;
      logitAndId[i][0] = s;
    }
    let r = Math.random() * sumS;
    for (let i = 0; i < logitAndId.length; i++) {
      r -= logitAndId[i][0];
      if (r <= 0) {
        return logitAndId[i][1];
      }
    }
    return logitAndId[0][1];
  }
}
