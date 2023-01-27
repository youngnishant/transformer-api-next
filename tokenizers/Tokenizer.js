"use strict";
import { TokenProcessor } from "./TokenProcessor.js";
import { TokenLattice } from "./TokenLattice.js";
import { CharTrie } from "./CharTrie.js";

export async function loadTokenizer(url) {
  const response = await fetch(url);
  const jtokenizer = await response.json();
  return Tokenizer.fromConfig(jtokenizer);
}

export class Tokenizer {
  bosToken;
  bosTokenId;
  eosToken;
  eosTokenId;
  unkToken;
  unkTokenId;
  vocab;
  specialTokens;
  specialTokenIds;
  trie;
  tokenToIds;
  minScore;
  unkScore;
  preTokenizer;
  normalizer;
  decoder;
  constructor(
    vocab,
    unkTokenId,
    specialTokens,
    normalizer,
    preTokenizer,
    decoder
  ) {
    this.vocab = vocab;
    this.unkTokenId = unkTokenId;
    this.specialTokens = specialTokens;
    this.specialTokenIds = new Map(specialTokens.map((x) => [x.id, x]));
    this.normalizer = normalizer;
    this.preTokenizer = preTokenizer;
    this.decoder = decoder;
    this.tokenToIds = new Map(
      vocab.map((x, i) => [this.normalize(String(x[0])), i])
    );
    this.bosToken = this.normalize(" ");
    this.bosTokenId = this.getTokenId(this.bosToken);
    this.eosToken = "</s>";
    this.eosTokenId = this.getTokenId(this.eosToken);
    this.unkToken = String(this.vocab[this.unkTokenId][0]);
    this.trie = new CharTrie();
    this.minScore = 1.0e6;
    vocab.forEach(
      (x) => (this.minScore = Math.min(this.minScore, Number(x[1])))
    );
    this.unkScore = this.minScore - 10.0;
    vocab[unkTokenId][1] = this.unkScore;
    vocab.forEach((x) => this.trie.push(String(x[0])));
  }
  static fromConfig(config) {
    const preTokenizer = TokenProcessor.fromConfig(config.pre_tokenizer);
    const normalizer = TokenProcessor.fromConfig(config.normalizer);
    const decoder = TokenProcessor.fromConfig(config.decoder);
    return new Tokenizer(
      config.model.vocab,
      config.model.unk_id,
      config.added_tokens,
      normalizer,
      preTokenizer,
      decoder
    );
  }
  getTokenId(normalizedToken) {
    return this.tokenToIds.get(normalizedToken) || this.unkTokenId;
  }

  normalize(text) {
    return this.normalizer.normalize(text);
  }
  preTokenize(normalized) {
    return this.preTokenizer.preTokenize(normalized);
  }
  populateNodes(lattice) {
    const sentence = lattice.sentence;
    const len = sentence.length;
    let beginPos = 0;
    while (beginPos < len) {
      const mblen = 1;
      let hasSingleNode = false;
      const tokens = [];
      for (const token of this.trie.commonPrefixSearch(
        sentence.slice(beginPos)
      )) {
        tokens.push(token);
        const tokenId = this.getTokenId(token);
        const tokenScore = Number(this.vocab[tokenId][1]);
        const n = token.length;
        lattice.insert(beginPos, n, tokenScore, tokenId);
        if (!hasSingleNode && n == mblen) {
          hasSingleNode = true;
        }
      }
      if (!hasSingleNode) {
        lattice.insert(beginPos, mblen, this.unkScore, this.unkTokenId);
      }
      beginPos += mblen;
    }
  }
  tokenize(normalized) {
    const lattice = new TokenLattice(
      normalized,
      this.bosTokenId,
      this.eosTokenId
    );
    this.populateNodes(lattice);
    const tokenIds = lattice.tokenIds();
    return tokenIds;
  }
  encode(text) {
    if (text === null || text === undefined || text.length === 0)
      return [this.eosTokenId];
    const normalized = this.normalize(text);
    const pre = this.preTokenize([normalized]);
    const tokens = [];
    for (const token of pre) {
      const tokenized = this.tokenize(token);
      tokens.push(...tokenized);
    }
    tokens.push(this.eosTokenId);
    return tokens;
  }
  decode(tokenIds, skipSpecialTokens) {
    const tokens = tokenIds.map((x) => {
      if (this.specialTokenIds.get(x) !== undefined && skipSpecialTokens) {
        return "";
      } else if (x == this.unkTokenId) {
        return this.unkToken + " ";
      } else if (x in this.vocab) {
        return String(this.vocab[x][0]);
      } else {
        return `[${x}]`;
      }
    });
    const decodedTokens = this.decoder.decodeChain(tokens);
    const decoded = decodedTokens.join("");
    return decoded;
  }
}
