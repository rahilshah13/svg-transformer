1. `docker compose up -d --build`
2. `cp *.jpg ./watch/*.jpg; cp *.png ./watch/*.png`
2. `find . -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) -exec mv {} ./watch/ \;`
3. `localhost:8000`