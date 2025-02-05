
import fs from 'fs';
import puppeteer from 'puppeteer';


/*
const { readFileSync, writeFile } = await import('fs');
const puppeteer = await import('puppeteer');
*/


const outfile = 'scrape/desuarchive_msdg_20240331.json';
// const startTarget = 'https://desuarchive.org/trash/thread/64446742';
// Jan 2024
// const startTarget = 'https://desuarchive.org/trash/thread/62548034';
const startTarget = 'https://desuarchive.org/trash/thread/62548034';

(async () => {

  console.log('Starting puppeteer.');
  let browser = await puppeteer.launch();
  console.log('Opening tab.');
  let page = await browser.newPage();


  console.log('Disabling images.');
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36');
  // No image dl
  await page.setRequestInterception(true);
  page.on('request', (req) => { if(req.resourceType() === 'image') { req.abort(); } else { req.continue(); } });



  let textData;
  try {
    textData = await fs.promises.readFile(outfile, 'utf8');
    textData = JSON.parse(textData);
  }
  catch (err) {
    console.log('No file found, starting fresh.');
    textData = {};
  }

  let currentUrl = startTarget;

  // TODO: Grab text desc too?
  for (let pageCount = 0; pageCount < 20; pageCount++) {
    console.log(`Browsing to link ${pageCount}/20 - ${currentUrl}`);
    await page.goto(currentUrl);
    console.log('Page loaded.');

    let page_links = await page.$$eval('a.post_file_filename', (elements) => elements.map(el => {return {link: el.href, filename: el.innerText}}))
    console.log(`found ${page_links.length} links.`);

    let page_id = currentUrl.split('/').slice(-1)[0];
    textData[page_id] = page_links;

    currentUrl = await page.$eval('::-p-text(Previous thread) a', el => el.href);
  }
  await browser.close();

  console.log(`Next URL would be: ${currentUrl}`);

  try {
    await fs.promises.writeFile(outfile, JSON.stringify(textData, null, "  "));
    console.log('The file has been saved!');
  } catch (err) {
    console.log('The file could not be written.', err)
  };

  console.log('Done.');
})();