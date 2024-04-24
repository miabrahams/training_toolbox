

import {promises as fsPromises} from 'fs';
import { exit } from 'process';
import puppeteer from 'puppeteer';



// FA
const fa_options = {
    cookies_file: "www.furaffinity.net.cookies.json",
    target_url: "https://www.furaffinity.net/gallery/claweddays/",
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
  return await page.$$eval('div.post', (posts) =>
    posts.map((post) => {
      return {
        author: post.querySelector('.author strong').innerText,
        id: post.id,
        content: post.querySelector('.content').innerText,
      };
    })
  );
}

async function ar_labelVisitor(page, url, pageData, allData) {
  // Just '1'
  const pageNumber = page.$eval('.pagination strong').innerText;
  const pageLabel = `page${pageNumber}`
  allData[pageLabel] = pageData;
}


async function ar_itemVisitor(page, link, textData, currentUrl = null) {
  return null;
}

const ar_nextLinkSelector = 'div.pagination > span > a:nth-of-type(1)';

const config = {
  cookies_file: null,
  target_url: "https://aryion.com/forum/viewtopic.php?f=18&t=457",
  output: "aryion.json",
  max_pages: 1,
  labelVisitor: ar_labelVisitor,
  pageVisitor: ar_pageVisitor,
  nextLinkSelector: ar_nextLinkSelector,
  itemVisitor: ar_itemVisitor
};


// pageVisitor: Given current page, return an array of objects containing target data.
// labelVisitor: Store data from current page into the output data object. Could look up page num, URL, etc
// itemVisitor: do something on each data item. Not necessary for eg forum threads.



(async () => {
  const {target_url, maxPages, pageVisitor, labelVisitor, nextLinkSelector, itemVisitor, cookies_file} = config;

  let cookies = null;
  if (cookies_file) {
    cookies = await fsPromises.readFile(`data/cookies/${config.cookies_file}`).then(JSON.parse);
  }


  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  if (cookies) { await page.setCookie(...cookies); }



  // Use existing file
  const outfile = `data/scrape/${config.output}`
  let data;
  try {
    data = await fs.promises.readFile(outfile, 'utf8').then(JSON.parse);
  }
  catch (err) {
    console.log('No existing data found. Creating new file.');
    data = {};
  }



  // Disable image loading
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setRequestInterception(true);
  page.on('request', (req) => { if (req.resourceType() === 'image') req.abort(); else req.continue(); });


  // Start here
  let currentUrl = target_url;


  for (let pageCount = 0; pageCount < maxPages; pageCount++) {
    if (!currentUrl) break;
    await page.goto(currentUrl);
    console.log(`Loaded page ${pageCount}/${maxPages} - ${currentUrl}`);
    let pageData = await pageVisitor(page);
    await labelVisitor(page, currentUrl, data);

    // Find next page
    currentUrl = await page.$eval(nextLinkSelector, el => el.href);
  }



  /* TODO: Visit each link individually (For gallery scraping)
  for (const i in links) {
    console.log(`Visiting link ${i}/${links.length}`);
    visitor(links[i], textData);
  }
  */

  await browser.close();

  try {
    await fsPromises.writeFile(
      outfile,
      JSON.stringify(data, null, '  ')
    );
  } catch (err) {
    console.log('The file could not be written.', err);
  }

  console.log('Done.');
})();
