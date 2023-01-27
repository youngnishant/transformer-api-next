export class Seq2SeqLMOutput {
  logits;
  pastKeyValues;
  encoderOutputs;
  constructor(logits, pastKeyValues, encoderOutputs) {
    this.logits = logits;
    this.pastKeyValues = pastKeyValues;
    this.encoderOutputs = encoderOutputs;
  }
}
