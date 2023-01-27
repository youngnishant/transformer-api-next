"use strict";
import { Tokenizer } from "./Tokenizer.js";
// import fetch from "node-fetch";
import tokenizerJson from "../models/t5-small-tokenizer.json" assert { type: "json" };
export class AutoTokenizer {
  tokenizer = null;
  modelId;
  modelsPath;

  constructor(modelId, modelsPath) {
    this.modelId = modelId;
    this.modelsPath = modelsPath;
  }

  static fromPretrained(modelId, modelsPath) {
    return new AutoTokenizer(modelId, modelsPath);
  }

  async load() {
    if (this.tokenizer != null) {
      return this.tokenizer;
    }
    const modelIdParts = this.modelId.split("/");
    const modelName = modelIdParts[modelIdParts.length - 1];
    // const url = `${this.modelsPath}/${modelName}-tokenizer.json`;
    // const response = await fetch(url);
    this.tokenizer = Tokenizer.fromConfig(tokenizerJson);
    return this.tokenizer;
  }

  async encode(text) {
    const tokenizer = await this.load();
    return tokenizer.encode(text);
  }

  async decode(tokenIds, skipSpecialTokens) {
    const tokenizer = await this.load();
    return tokenizer.decode(tokenIds, skipSpecialTokens);
  }
}
