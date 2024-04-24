
import { readFileSync, writeFileSync } from 'fs';
import puppeteer from 'puppeteer';

const userdata = JSON.parse(readFileSync('cookies/userdata.json', 'utf8'));

// Ain't working tbh


(async () => {
    console.log('starting.');
    const browser = await puppeteer.launch();
    console.log('Puppeteer started.');
    const page = await browser.newPage();
    console.log('Opening tab.');
    await page.goto('https://www.furaffinity.net/login');
    console.log('Loading FA');

    // Login
    // await page.type('#login', userdata.username);
    // await page.type('input[name="pass"]', userdata.password);
    await page.type('#login', 'test');
    await page.type('input[name="pass"]', 'test2');
    // await page.click('#login-button');
    await page.keyboard.press('Enter',{delay:2000});

    const loginB = await page.$eval('#login-button', el => el.outerHTML);
    const pass = await page.$eval('input[name="pass"]', el => el.outerHTML);
    const loginF = await page.$eval('#login', el => el.outerHTML);

    console.log(loginB);
    console.log(pass);
    console.log(loginF);

    console.log('Did the things.');
    await page.waitForNavigation();
    console.log('Awaiting navigation.');

    // Get cookies
    const cookies = await page.cookies();

/*     // Use cookies in other tab or browser
    const page2 = await browser.newPage();
    await page2.setCookie(...cookies);
    await page2.goto('https://facebook.com'); // Opens page as logged user
 */

    // Write cookies to temp file to be used in other profile pages
    cookiesFilePath = './cookies/FA.json'
    writeFileSync(cookiesFilePath, cookies, JSON.stringify(cookies),
      function(err) {
        if (err) {
          console.log('The file could not be written.', err)
        }
        console.log('Session has been successfully saved')
      })


    await browser.close();

    console.log('Done.');
})();