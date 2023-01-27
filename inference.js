// export { Tokenizer } from "../tokenizers/Tokenizer";
import { T5ForConditionalGeneration } from "./transformers/T5ForConditionalGeneration.js";
import { AutoTokenizer } from "./tokenizers/AutoTokenizer.js";
import path from "path";

const generate = async () => {
  const command = {
    inputText: "translate English to French: The universe is a dark forest",
    modelId: "t5-small",
    modelsPath: path.resolve() + "/models",
    maxLength: 50,
    topK: 0,
  };
  const modelId = command.modelId;
  const modelsPath = command.modelsPath;

  const model = new T5ForConditionalGeneration(modelId, modelsPath);
  const tokenizer = AutoTokenizer.fromPretrained(modelId, modelsPath);

  const inputText = command.inputText;
  const inputTokenIds = await tokenizer.encode(inputText);
  const generationOptions = {
    maxLength: command.maxLength,
    topK: command.topK,
  };
  // function delayMillis(millis) {
  //   return new Promise((resolve) => {
  //     setTimeout(() => {
  //       resolve(0);
  //     }, millis);
  //   });
  // }
  // async function generateProgress(outputTokenIds) {
  // const outputText = (await tokenizer.decode(outputTokenIds, true)).trim();
  // console.log({
  //   inputText: inputText,
  //   outputText: outputText,
  //   complete: false,
  // });
  // await delayMillis(1);
  //   return true;
  // }
  const finalOutputTokenIds = await model.generate(
    inputTokenIds,
    generationOptions
    // generateProgress
  );
  const finalOutput = (
    await tokenizer.decode(finalOutputTokenIds, true)
  ).trim();
  return {
    inputText: inputText,
    outputText: finalOutput,
    // complete: true,
  };
};

export default generate;
// generate();
