urls = open("url_sql_dump.tsv", encoding="latin-1").readlines()
urls = [url.split("\t") for url in urls]
url_map = set()
for url in urls[1:]:
    if "vogue" in url[1]:
        url_map.add(url[0])

lines = open("text_sql_dump.tsv", encoding="latin-1").readlines()
lines = [line.strip().split("\t") for line in lines]
prev = []
mod_lines = []
for line in lines:
    if len(line) == len(fields):
        if prev:
            _id = prev[0][0]
            text = prev[0][1].strip('"')
            for x in prev[1:]:
                text += "<NEW_LINE>" + x[0].strip('"')
            fin_ans = prev[-1]
            fin_ans[0] = text.replace("\x92", "'")
            fin_ans.insert(0, _id)
            mod_lines.append(fin_ans[:])
            prev = []

        mod_lines.append(line[:])
        continue
    prev.append(line[:])

for line in mod_lines:
    if line[2] == '"BODY"' and line[7] in url_map:
        x = line[1].replace("<NEW_LINE>", " ").strip('"') + "\n"
        x = "".join([ch for ch in x if ch in string.printable])
        data += x