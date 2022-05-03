set IMAGE_URI=gcr.io/your-projectid-123456/your_image:your_tag
docker build -f dockerfile -t %IMAGE_URI% ./