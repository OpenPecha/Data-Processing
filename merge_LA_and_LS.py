from pathlib import Path
import jsonlines

def write_jsonl(final_jsonl, jsonl_path):
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write_all(final_jsonl)


def get_spans(spans):
    final_span = []
    for span in spans:
        try:
            label = span["label"]
        except:
            continue
        span = {
            "label": label,
            "points": span["points"]
        }
        final_span.append(span)
    return final_span


def get_width_and_height(line):
    try:
        width = line["width"]
        height = line["height"]
        return width, height
    except:
        return None, None


def filter_LS_jsonl(LS_path):
    LS_dict = {}
    curr = {}
    with jsonlines.open(LS_path) as reader:
        for line in reader:
            if line["answer"] == "accept":
                if line["spans"]:
                    image_name = line['id'].split(".")[0]
                    width, height = get_width_and_height(line)
                    curr[image_name] = {
                        "id": image_name,
                        "image": line["image"],
                        "width": width,
                        "height": height,
                        "spans": get_spans(line["spans"])
                    }
                    LS_dict.update(curr)
                    curr = {}
    return LS_dict

def merge_jsonl(LA_path, LS_path):
    merged_western_jsonl = []
    merged_perig_jsonl = []
    LS_dict = filter_LS_jsonl(LS_path)
    with jsonlines.open(LA_path) as reader:
        for line in reader:
            image_name = line["id"].split(".")[0]
            image_dict = LS_dict.get(image_name, None)
            if image_dict:
                if line["width"] == None:
                    line["width"] = image_dict["height"]
                    line["height"] = image_dict["height"]
                line["id"] = image_dict["image"].split("/")[-1].split("?")[0]
                line["spans"].extend(image_dict["spans"])
                if line["width"] < line["height"]:
                    merged_western_jsonl.append(line)
                else:
                    merged_perig_jsonl.append(line)
            else:
                print(f"{image_name} not found in LS_jsonl.jsonl")
                continue
            LS_dict.pop(image_name)
    return merged_western_jsonl, merged_perig_jsonl


if __name__ == "__main__":
    LA_path = Path("./data/LA_jsonl.jsonl")
    LS_path = Path("./data/LS_jsonl.jsonl/")
    merged_western_jsonl, merged_perig_jsonl = merge_jsonl(LA_path, LS_path)
    write_jsonl(merged_western_jsonl, "./data/merged_western_jsonl.jsonl")
    write_jsonl(merged_perig_jsonl, "./data/merged_perig_jsonl.jsonl")