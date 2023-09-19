import jsonlines
import xml.etree.ElementTree as ET
from pathlib import Path


def get_json_coordinates(points_str):
    return [list(map(float, point.split(','))) for point in points_str.split()]

def xml_to_dict(xml_file):
    # Parse the XML data
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image = xml_file.stem
    except Exception as e:
        print(f"Error parsing XML: {xml_file}")
        return {}

    ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # Check if 'Page' key exists in the dictionary
    page_element = root.find('ns:Page', ns)
    if page_element is None:
        print(f"'Page' element not found in XML: {xml_file}")
        return

    # Extract the necessary fields to create the JSONL record
    json_record = {
        "image": f"{image}.jpg",
        "width": int(page_element.get("imageWidth", 0)),
        "height": int(page_element.get("imageHeight", 0)),
        "spans": []
    }

    # Handle TextRegion elements
    for region in page_element.findall('ns:TextRegion', ns):
        span = {
            "label": None,
            "points": get_json_coordinates(region.find('ns:Coords', ns).get('points'))
        }
        custom = region.get("custom", "")
        if "marginalia" in custom:
            span["label"] = "Margin"
        elif "caption" in custom:
            span["label"] = "Caption"
        elif "header" in custom:
            span["label"] = "Header"
        elif "footer" in custom:
            span["label"] = "Footer"
        else:
            span["label"] = "Text-Area"
        json_record["spans"].append(span)

    # Handle ImageRegion elements
    for region in page_element.findall('ns:ImageRegion', ns):
        span = {
            "label": "Illustration",
            "points": get_json_coordinates(region.find('ns:Coords', ns).get('points'))
        }
        json_record["spans"].append(span)

    return json_record

def write_jsonl(final_jsonl, jsonl_path):
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write_all(final_jsonl)

def main():
    jsonls = []
    jsonl_out = Path(f"./data/LA_jsonl.jsonl")
    xml_paths = list(Path(f"./data/page/").iterdir())
    for xml_path in xml_paths:
        xml_dict = xml_to_dict(xml_path)
        if xml_dict:
            jsonls.append(xml_dict)
    write_jsonl(jsonls, jsonl_out)

if __name__ == '__main__':
    main()
