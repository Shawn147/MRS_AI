// Check the real browser bridge without touching a user's chat or browser.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const script = fs.readFileSync('assets/browser_history/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const frames = [], events = {}, docEvents = {}, calls = [];
const main = {scrollHeight: 3000, clientHeight: 700, scrollTop: 0,
  getBoundingClientRect: () => ({width: 900, right: 1000}),
  scrollTo({top}) {this.scrollTop = Math.min(top, this.scrollHeight - this.clientHeight); calls.push(this.scrollTop);}};
const message = {offsetHeight: 200, getBoundingClientRect: () => ({bottom: 1500})};
const composer = {getBoundingClientRect: () => ({height: 100, top: 900})};
const doc = {querySelector: () => main, querySelectorAll: () => [composer],
  getElementById: () => ({closest: () => message}), documentElement: {style: {setProperty() {}}},
  addEventListener(name, handler) {docEvents[name] = handler;},
  removeEventListener(name, handler) {if (docEvents[name] === handler) delete docEvents[name];}};
const host = {document: doc, innerHeight: 1000, innerWidth: 1000,
  getComputedStyle: () => ({display: 'block'}),
  ResizeObserver: class {observe() {} unobserve() {} disconnect() {}},
  addEventListener() {}, removeEventListener() {}, postMessage() {},
  requestAnimationFrame(fn) {frames.push(fn);}};
const window = {parent: host, addEventListener(name, fn) {events[name] = fn;}};
vm.runInNewContext(script, {window});
const render = request => events.message({source: host, data: {type: 'streamlit:render',
  args: {action: 'scroll', anchor: 'chat-message-' + 'a'.repeat(32), request_id: request}}});
render('first');
frames.shift()();
assert.equal(main.scrollTop, 2300, 'new prompt scrolls to the end, not only its message edge');
main.scrollHeight = 3600;
while (frames.length) frames.shift()();
assert.equal(main.scrollTop, 2900, 'later content and follow-up controls stay visible');
const count = calls.length;
render('first');
assert.equal(frames.length, 0, 'same rerun request does not pull the user back down');
render('second');
frames.shift()();
docEvents.wheel();
main.scrollHeight = 4000;
while (frames.length) frames.shift()();
assert.equal(calls.length, count + 1, 'user scrolling interrupts auto-scroll');
console.log('Chat scroll checks passed');
