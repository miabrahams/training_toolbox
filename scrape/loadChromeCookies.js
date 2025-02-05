import { getCookies as _getCookies } from 'chrome-cookies-secure';
import { writeFileSync } from 'fs';

const url = 'https://www.furaffinity.net';


const cookiesFilePath = 'cookies/FA_Default.json';
const profile ='Default'; // e.g. 'Profile 2'

// find profiles at ~/Library/Application Support/Google/Chrome
_getCookies(url, 'puppeteer', function(err, cookies) {
    if (err) {
        console.log(err, 'error');
        return
    }
    console.log("Got cookies: ", cookies);
    writeFileSync(cookiesFilePath, JSON.stringify(cookies),
      function(err) {
        if (err) {
          console.log('The file could not be written.', err)
        }
        console.log('Session has been successfully saved')
      })
  }, profile);
