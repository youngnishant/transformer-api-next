"use strict";

export class TokenProcessor {
  static fromConfig(config) {
    switch (config.type) {
      case "Metaspace":
        return new MetaspaceTokenProcessor(
          config.add_prefix_space || false,
          config.replacement || "",
          config.str_rep
        );
      case "Precompiled":
        return new PrecompiledTokenProcessor(config.precompiled_charsmap);
      case "Sequence":
        return new SequenceTokenProcessor(
          (config.pretokenizers || []).map((x) => TokenProcessor.fromConfig(x))
        );
      case "WhitespaceSplit":
        return new WhitespaceSplitTokenProcessor();
      default:
        throw new Error("Unknown token processor type: " + config.type);
    }
  }
  normalize(text) {
    return text;
  }
  preTokenize(tokens) {
    return tokens;
  }
  decodeChain(tokens) {
    return tokens;
  }
}

class MetaspaceTokenProcessor extends TokenProcessor {
  addPrefixSpace;
  replacement;
  strRep;
  constructor(add_prefix_space, replacement, str_rep) {
    super();
    this.addPrefixSpace = add_prefix_space;
    this.replacement = replacement;
    this.strRep = str_rep || this.replacement;
  }
  preTokenize(normalizedTokens) {
    const result = [];
    for (const token of normalizedTokens) {
      let normalized = token.replace(" ", this.strRep);
      if (this.addPrefixSpace && !normalized.startsWith(this.replacement)) {
        normalized = this.strRep + normalized;
      }
      result.push(normalized);
    }
    return result;
  }
  decodeChain(tokens) {
    const result = [];
    let i = 0;
    for (const token of tokens) {
      let normalized = token.replace(this.replacement, " ");
      if (this.addPrefixSpace && i == 0 && normalized.startsWith(" ")) {
        normalized = normalized.substring(1);
      }
      result.push(normalized);
      i++;
    }
    return result;
  }
}

class PrecompiledTokenProcessor extends TokenProcessor {
  charsmap;
  constructor(charsmap) {
    super();
    this.charsmap = charsmap;
  }
  normalize(text) {
    return text;
  }
}

class SequenceTokenProcessor extends TokenProcessor {
  tokenizers;
  constructor(tokenizers) {
    super();
    this.tokenizers = tokenizers;
  }
  preTokenize(normalizedTokens) {
    let result = normalizedTokens;
    for (const tokenizer of this.tokenizers) {
      result = tokenizer.preTokenize(result);
    }
    return result;
  }
}

class WhitespaceSplitTokenProcessor extends TokenProcessor {
  preTokenize(normalizedTokens) {
    const result = [];
    for (const token of normalizedTokens) {
      result.push(...token.split(/\s+/));
    }
    return result;
  }
}
