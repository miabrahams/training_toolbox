import * as fs from 'fs';
import * as path from 'path';

// Function to recursively get all filenames with .jpg or .png extensions
function getImageFilenames(directory) {
  const filenames = [];

  function traverseDirectory(currentDirectory) {
    const files = fs.readdirSync(currentDirectory);
    files.forEach((file) => {
      const filePath = path.join(currentDirectory, file);
      const stats = fs.statSync(filePath);
      if (stats.isDirectory()) {
        traverseDirectory(filePath);
      } else if (stats.isFile() && ['.jpg', '.png'].includes(path.extname(file).toLowerCase())) {
        filenames.push(file);
      }
    });
  }

  traverseDirectory(directory);
  return filenames;
}

// Specify the directory to scan for image files
const directory = 'D:/Sync_AI/Training/4chan';

// Get all image filenames
const imageFilenames = getImageFilenames(directory);


// Save the filenames to a JSON file
fs.writeFileSync('scrape/existing_filenames.json', JSON.stringify(imageFilenames));

console.log('Filenames saved to filenames.json');