"use strict";

export class CharTrie {
  root;
  constructor() {
    this.root = CharTrieNode.default();
  }
  push(text) {
    let node = this.root;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      let child = node.children.get(ch);
      if (child === undefined) {
        child = CharTrieNode.default();
        node.children.set(ch, child);
      }
      node = child;
    }
    node.isLeaf = true;
  }
  *commonPrefixSearch(text) {
    let node = this.root;
    let prefix = "";
    for (let i = 0; i < text.length && node !== undefined; i++) {
      const ch = text[i];
      prefix += ch;
      node = node.children.get(ch);
      if (node !== undefined && node.isLeaf) {
        yield prefix;
      }
    }
  }
}

class CharTrieNode {
  isLeaf;
  children;
  constructor(isLeaf, children) {
    this.isLeaf = isLeaf;
    this.children = children;
  }
  static default() {
    return new CharTrieNode(false, new Map());
  }
}
