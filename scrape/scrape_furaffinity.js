
import { readFileSync, writeFile } from 'fs';
import puppeteer from 'puppeteer';

const cookies = JSON.parse(readFileSync('cookies/www.furaffinity.net.cookies.json', 'utf8'));
// const target = 'https://www.furaffinity.net/gallery/foxyguts/';
// const out = 'scrape/foxyguts.json';

const target = 'https://www.furaffinity.net/gallery/claweddays/';
const out = 'scrape/claweddays.json';


(async () => {
  console.log('Starting puppeteer.');
  const browser = await puppeteer.launch();
  console.log('Opening tab.');
  const page = await browser.newPage();


  console.log('Disabling images.')
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setRequestInterception(true);
  page.on('request', (req) => {
      if(req.resourceType() === 'image'){
          req.abort();
      }
      else {
          req.continue();
      }
  });

  console.log('Setting cookies.');
  await page.setCookie(...cookies);

  console.log('Loading FA.');
  await page.goto(target);
  const links = await page.$$eval('#gallery-gallery > figure > p:nth-child(1) > a', links => links.map(a => [a.href, a.innerText]));
  console.log(links);


  // TODO: Grab titles as well as links!!
  const textData = {};
  for(const i in links) {
    console.log(`Browsing to link ${i}/${links.length}`);
    try {
      await page.goto(links[i][0]);
      const text = await page.$eval('.submission-description', desc => desc.innerText)
      textData[links[i][1]] = text;
    }
    catch (err) {
      console.log("Error: ", err)
    }
  }

  await browser.close();

  console.log("Scraped data: ", textData)

  try {
    writeFile(out, JSON.stringify(textData, null, "  "), (err) => {
      if (err) throw err;
      console.log('The file has been saved!');
    });
  } catch (err) {
      console.log('The file could not be written.', err)
  };

  console.log('Done.');
})();