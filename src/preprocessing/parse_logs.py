import re
from datetime import datetime


def extract_timestamp(message):
    match = re.search(r"(\w{3}) (\d{1,2}) (\d{2}:\d{2}:\d{2})", message)
    if match:
        month_str = match.group(1)
        day = match.group(2)
        time = match.group(3)

        month_map = {
            "Jan": "01", "Feb": "02", "Mar": "03",
            "Apr": "04", "May": "05", "Jun": "06",
            "Jul": "07", "Aug": "08", "Sep": "09",
            "Oct": "10", "Nov": "11", "Dec": "12"
        }

        month = month_map.get(month_str, "01")

        # build full timestamp
        return f"2024-{month}-{day.zfill(2)} {time}"

    return None


def extract_ip(message):
    match = re.search(r"\[(\d{1,3}(?:\.\d{1,3}){3})\]", message)
    if match:
        return match.group(1)
    return ""


def parse_logs(input_file, output_file):
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(output_file, "w") as f_out:

        f_out.write("timestamp,ip,message\n")

        for line in f_in:
            line = line.strip()

            if not line:
                continue

            timestamp = extract_timestamp(line)
            ip = extract_ip(line)

            # clean message
            message = line.replace(",", " ")

            if timestamp:
                f_out.write(f"{timestamp},{ip},\"{message}\"\n")


if __name__ == "__main__":
    input_file = "data/split/zimbra_24_train.log"
    output_file = "outputs/parsed_logs.csv"

    parse_logs(input_file, output_file)

    print("parsing done")