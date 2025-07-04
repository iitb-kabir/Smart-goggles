#Install the  Python packages
##==================================
# !pip install bing-image-downloader
# !pip install pillow
##==================================

# Import the packages
from bing_image_downloader import downloader
import os

# Celebrity List

footballers = [
    "Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland", "Kevin De Bruyne",
    "Neymar", "Mohamed Salah", "Harry Kane", "Robert Lewandowski", "Vinícius Júnior",
    "Luka Modrić", "Sadio Mané", "Karim Benzema", "Antoine Griezmann", "Paulo Dybala",
    "Romelu Lukaku", "Phil Foden", "Joshua Kimmich", "Son Heung-min", "Bruno Fernandes",
    "Jude Bellingham", "Marcus Rashford", "Pedri", "João Félix", "Virgil van Dijk"
]

marvel_actors = [
    "Robert Downey Jr.", "Chris Evans", "Scarlett Johansson", "Chris Hemsworth", "Mark Ruffalo",
    "Tom Holland", "Benedict Cumberbatch", "Elizabeth Olsen", "Paul Rudd", "Brie Larson",
    "Jeremy Renner", "Don Cheadle", "Sebastian Stan", "Anthony Mackie", "Tom Hiddleston",
    "Zoe Saldaña", "Dave Bautista", "Karen Gillan", "Chris Pratt", "Samuel L. Jackson",
    "Gwyneth Paltrow", "Hayley Atwell", "Letitia Wright", "Danai Gurira", "Florence Pugh"
]
## combine both the list
all_celebrities = footballers + marvel_actors  # Total: 50


# Ensure your base dataset directory exists
base_dir = "/content/face_dataset"
os.makedirs(base_dir, exist_ok=True)

for name in all_celebrities:
    # Clean folder name
    folder_name = name.replace(" ", "_")
    save_dir = os.path.join(base_dir, folder_name)
    # Download images
    downloader.download(
        name,
        limit=25,
        output_dir=base_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
    print(f"Downloaded images for {name}")
