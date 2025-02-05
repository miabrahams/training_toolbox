import fs from 'fs';
import path from 'path';
import https from 'https';


// Read the JSON file
const jsonData = JSON.parse(fs.readFileSync('scrape/desuarchive_msdg_20240331.json', 'utf-8'));

// Set to store already downloaded files.
// Populate with getExistingFiles.js
const savedFilenames = JSON.parse(fs.readFileSync('scrape/existing_filenames.json', 'utf-8'));


// If we run this script multiple times, we may have already downloaded some files.
// We will re-create the set of unique filenames on every run.
const uniqueFilenames = new Set(savedFilenames);

// Track all files that have been downloaded at any point.
const downloadedFilenames = fs.readdirSync('downloaded_files');
const alreadyDownloaded = new Set(downloadedFilenames.concat(savedFilenames));

console.log(`Found ${alreadyDownloaded.size} existing files.`);

// Function to download an image.
// Desuarchive is tricky and sends a successful response, but it's just a text string.
function downloadImage(url, filename, callback) {
  https.get(url, (response) => {
    response.setEncoding('binary');
    let chunks = [];

    response.on('data', (chunk) => { chunks.push(Buffer.from(chunk, 'binary')); });
    response.on('end', () => {
      const binary = Buffer.concat(chunks);
      const fileData = binary.toString('binary');

      if (fileData.includes('error code: 1015')) {
        console.error(`Error downloading ${filename}: Throttled.`);
        downloadQueue.unshift({ link:url, filename });
        setTimeout(callback, 2000);
      } else {
        fs.writeFile(`downloaded_files/${filename}`, binary, 'binary', (err) => {
          if (err) console.error(`Error saving file ${filename}: ${err.message}`);
          else console.log(`Downloaded ${filename}. (${downloadQueue.length} files remaining)`);
          setTimeout(callback, 2000);
        });
      }
    });
  }).on('error', (err) => {
    console.error(`Error downloading ${url}: ${err.message}`);
    setTimeout(callback, 1000);
  });
}



// Create a queue of download tasks
const downloadQueue = [];
const catboxQueue = [];
let nSkipped = 0;

// Function to process the next download task using a 2.01-second delay
function processNextDownload() {
  if (downloadQueue.length > 0) {
    const { link, filename } = downloadQueue.shift();
    const startTime = Date.now();

    downloadImage(link, filename, () => {
      const endTime = Date.now();
      const elapsedTime = endTime - startTime;
      const waitTime = Math.max(0, 2010 - elapsedTime);
      setTimeout(processNextDownload, waitTime);
    })
  }
}



// Iterate over each page in the JSON data
for (const pageNumber in jsonData) {
  const pageData = jsonData[pageNumber];

  // Iterate over each link in the page
  pageData.forEach((item) => {
    const link = item.link;
    let filename = item.filename;

    // Clean and sanitize for Windows
    filename = filename.replace(' (...)', '');
    filename = filename.replace(/[<>:"/\\|?*\x00-\x1F]/g, '');

    // Extract the numeric filename from the link URL
    const numericFilename = path.basename(link);


    // If numeric filename already downloaded, skip.
    if (alreadyDownloaded.has(numericFilename)) {
      nSkipped++;
      return;
    }

    // If filename is taken, use numeric filename as unique name.
    if (uniqueFilenames.has(filename)) {
      filename = numericFilename;
      uniqueFilenames.add(filename);
    }

    // Check if the file has already been downloaded
    if (alreadyDownloaded.has(filename)) {
      nSkipped++;
      return;
    }



    // Check if the filename is in the "catbox_xxxxxx.png" format
    const catboxMatch = filename.match(/^catbox_([\w\.]+)$/);
    if (catboxMatch) {
      const catboxUrl = `https://files.catbox.moe/${catboxMatch[1]}`;
      // downloadQueue.push({ link: catboxUrl, filename });
      catboxQueue.push({ link: catboxUrl, filename });
    } else {
      downloadQueue.push({ link, filename });
    }
  });
}

// Start processing the download queue
fs.writeFileSync('scrape/catbox_download_links.json', JSON.stringify(catboxQueue, null, "  "));
fs.writeFileSync('scrape/desuarchive_download_links.json', JSON.stringify(downloadQueue, null, "  "));
console.log('Saving queues.');



console.log(`Downloading ${downloadQueue.length} files.`);
console.log(`Skipped ${nSkipped} files already downloaded.`);

// Desuarchive seems to block cURL?
processNextDownload();