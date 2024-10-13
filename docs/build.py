from jinja2 import Environment, FileSystemLoader
from htmlmin.main import minify
import tomllib
import os
from os.path import join
import shutil
from sys import argv

target_folder = "./dist"

env = Environment(
    loader=FileSystemLoader("./template"),
    autoescape=False,
)

with open("./config.toml", 'rb') as config_file:
    config = tomllib.load(config_file)
    # pprint(config)

page = env.get_template('index.html').render(config)
page = minify(page, remove_comments = True, remove_all_empty_space=True)

if not os.path.isdir(target_folder):
    os.makedirs(target_folder, exist_ok=True)
else:
    os.system(f"rm -r '{target_folder}'")
    os.mkdir(target_folder)

with open(join(target_folder, "index.html"), 'w') as f:
    f.write(page)

shutil.copytree("./static", join(target_folder, "static"))

for key, file in config.get("static_file", {}).items():
    dest = join(target_folder, file['dest'])
    directory = os.path.dirname(dest)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if 'src' in file:
        shutil.copy(file['src'], dest)
    elif 'content' in file:
        content = file['content']
        with open(dest, 'w') as f:
            f.write(content)
print("done")