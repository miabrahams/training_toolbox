

import {promises as fsPromises} from 'fs';
import { exit } from 'process';
import puppeteer from 'puppeteer';



// FA
const fa_options = {
    cookies_file: "www.furaffinity.net.cookies.json",
    target_url: "https://www.furaffinity.net/gallery/honovy/",
    output: "claweddays.json"
}
const fa_selector = '#gallery-gallery > figure > p:nth-child(1) > a'
const fa_selectorFn = links => links.map(a => [a.href, a.innerText]);
const fa_nextLinkSelector = null;


// TODO: Grab titles as well as links!!
// Also: make visiting the links separate from finding them
async function fa_visitor(page, link, output, currentUrl = null) {
    try {
      await page.goto(link[0]);
      const text = await page.$eval('.submission-description', desc => desc.innerText)
      output[link[1]] = text;
    }
    catch (err) {
      console.log("Error: ", err)
    }
}


// Desuarchive
const ds_selector = 'a.post_file_filename';
const ds_selectorFn = (elements) => elements.map(el => {return {link: el.href, filename: el.innerText}});
const ds_nextLinkSelector = '::-p-text(Previous thread) a';
async function ds_visitor(page, link, textData, currentUrl = null){
  return null;
}

async function sd_labelVisitor(page, url) {
  page_id = url.split('/').slice(-1)[0];
}



// Aryion
async function ar_pageVisitor(page, url) {
  const data = await page.$$eval('div.post', (posts) =>
    posts.map((post) => {
      return {
        author: post.querySelector('.author strong').innerText,
        id: post.id,
        content: post.querySelector('.content').innerText,
      };
    })
  );

  const pageNumber = await page.$eval('.pagination strong', a => a.innerText);
  return {pageNumber, url, data};
}

const ar_nextLinkSelector = 'div.pagination > span > strong + .page-sep + a';

const config = {
  cookiesFile: null,
  targetUrl: "https://aryion.com/forum/viewtopic.php?f=18&t=457",
  outputName: "aryion2.json",
  maxPages: 570,
  pageVisitor: ar_pageVisitor,
  nextLinkSelector: ar_nextLinkSelector,
  itemVisitor: null,
  append: false
};


// pageVisitor: Given current page, return a data object to store in output array. Could also store page URL, etc if need be.
// itemVisitor: do something on each data item. Not necessary for eg forum threads.



(async () => {
  const {targetUrl, maxPages, pageVisitor, nextLinkSelector, itemVisitor, cookiesFile, outputName, append} = config;

  let cookies = null;
  if (cookiesFile) {
    cookies = await fsPromises.readFile(`data/cookies/${config.cookiesFile}`).then(JSON.parse);
  }


  console.log("Starting")
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  if (cookies) { await page.setCookie(...cookies); }



  // Use existing file
  const outFile = `data/scrape/${outputName}`;
  let data = [];

  if (append) {
    console.log("Looking for existing data.");
    try {
      data = await fs.promises.readFile(outfile, 'utf8').then(JSON.parse);
    }
    catch (err) {
      console.log('No existing data found. Creating new file.');
    }
  }


  // Disable image loading
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setRequestInterception(true);
  page.on('request', (req) => { if (req.resourceType() === 'image') req.abort(); else req.continue(); });


  // Start here
  let currentUrl = targetUrl;


  for (let pageCount = 0; pageCount < maxPages; pageCount++) {
    if (currentUrl === null) {
      console.log("Done.");
      break;
    }

    try {
      await page.goto(currentUrl);
      console.log(`Loaded page ${pageCount}/${maxPages} - ${currentUrl}`);
      const pageData = await pageVisitor(page, currentUrl);
      data.push(pageData);
      currentUrl = await page.$eval(nextLinkSelector, el => el.href);
    }
    catch (err) {
      console.error("Error: ", err);
      currentUrl = null;
    }
  }


  await browser.close();

  try {
    await fsPromises.writeFile(
      outFile,
      JSON.stringify(data, null, '  ')
    );
    console.log("Done.")
  } catch (err) {
    console.error('The file could not be written.', err);
    print(JSON.stringify(data));
  }
})();
