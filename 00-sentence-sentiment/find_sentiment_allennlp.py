from allennlp.predictors import Predictor
from tqdm import tqdm
import json
import argparse

def create_allennlp_sentiment_output(files, output_file, cuda_device=-1):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz", cuda_device=cuda_device)
    sentiment_output = []

    for file in tqdm(files, desc="file"):
        sentences = open(file).read().strip().split("\n")
        batch = [{"sentence": sentence} for sentence in sentences]
        output = predictor.predict_batch_json(batch)
        output = {"file": file, "sentiment": output}
        sentiment_output.append(output)

    json.dump(sentiment_output, open(output_file, "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use allennlp sentiment classifier to find 2-class sentiment of sentences", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", nargs="+", default=[f"/proj/sbaruah/elisa/bri/data/2020.11.24.homework/{x}.txt" for x in ["Heng","jon","peace","ulf"]], dest="files", help="list of txt files for which you want to find sentiment")
    parser.add_argument("--output", default="/proj/sbaruah/elisa/bri/allennlp_sentiment/data/sentiment.json", type=str, help="json file to which the output will be saved", dest="output")
    parser.add_argument("--device", default=-1, type=int, help="cuda device index; should be >= 0; CPU is used if not given", dest="device")
    
    args = parser.parse_args()
    files = args.files
    output_file = args.output
    device = args.device

    create_allennlp_sentiment_output(files, output_file, device)