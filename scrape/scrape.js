

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
  page_id = currentUrl.split('/').slice(-1)[0];
  textData[page_id] = page_links;
}


config = {
  cookies_file: null,
  target_url: "https://aryion.com/forum/viewtopic.php?f=18&t=457",
  output: ".json",
  selector: ds_selector,
  selectorFn: ds_selectorFn,
  nextLinkSelector: ds_nextLinkSelector,
  visitor: ds_visitor
};





(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  if (cookies) { await page.setCookie(...cookies); }


  // Use existing file
  const outfile = `data/scrape/${config.output}`
  let textData;
  try {
    textData = await fs.promises.readFile(outfile, 'utf8').then(JSON.parse);
  }
  catch (err) {
    console.log('No existing data found. Creating new file.');
    textData = {};
  }



  // Don't load images.
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setRequestInterception(true);
  page.on('request', (req) => { if (req.resourceType() === 'image') req.abort(); else req.continue(); });


  // Start browsing!
  let currentUrl = config.target_url;


  for (let pageCount = 0; pageCount < config.maxPages; pageCount++) {
    if (!currentUrl) break;
    await page.goto(currentUrl);
    console.log(`Loaded page ${pageCount}/${maxPages} - ${currentUrl}`);
    let page_links = await page.$$eval(selector, selectorFn);
    visitor(page, page_links, textData, currentUrl);
    currentUrl = await page.$eval(nextLinkSelector, el => el.href);
  }



  /* TODO: Visit each link individually (For FA style)
  for (const i in links) {
    console.log(`Visiting link ${i}/${links.length}`);
    visitor(links[i], textData);
  }
  */

  await browser.close();

  try {
    await fsPromises.writeFile(
      outfile,
      JSON.stringify(textData, null, '  ')
    );
  } catch (err) {
    console.log('The file could not be written.', err);
  }

  console.log('Done.');
})();
