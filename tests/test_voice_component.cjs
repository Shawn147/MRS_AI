// Exercise the browser component with simulated recognition, never a real microphone.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('assets/voice_input/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function setup(supported = true, missingFrameElement = false) {
  const messages = [], classes = new Set(), listeners = {}, elements = {};
  const makeClass = () => ({toggle(name, on) {on ? classes.add(name) : classes.delete(name);}, add(name) {classes.add(name);}, remove(name) {classes.delete(name);}});
  for (const id of ['mic', 'panel', 'status', 'preview', 'dot', 'stop', 'timer', 'close'])
    elements[id] = {style: {}, classList: makeClass(), textContent: '', disabled: false};
  const sendButton = {disabled: false};
  let instance;
  class Recognition {
    constructor() {instance = this;}
    start() {this.onstart();}
    stop() {this.onend();}
    abort() {this.onend?.();}
  }
  const window = {
    parent: {postMessage(message) {messages.push(message);}, document: {querySelector() {return sendButton;}, addEventListener() {}, removeEventListener() {}}},
    frameElement: {closest() {return {classList: makeClass()};}},
    isSecureContext: true, addEventListener(name, callback) {listeners[name] = callback;}
  };
  const hostContainer = {classList: makeClass()};
  const hostFrame = {contentWindow: window, closest() {return hostContainer;}};
  window.parent.document.querySelectorAll = () => [hostFrame];
  window.frameElement = missingFrameElement ? null : hostFrame;
  if (supported) window.SpeechRecognition = Recognition;
  const document = {body: {classList: makeClass()}, getElementById(id) {return elements[id];}, addEventListener() {}};
  vm.runInNewContext(script, {window, document, Date, Math, setTimeout() {}, setInterval() {}, clearTimeout() {}, clearInterval() {}});
  return {elements, messages, sendButton, classes, get instance() {return instance;},
    transcripts() {return messages.filter(m => m.type === 'streamlit:setComponentValue');}};
}
function result(text) {const value = [{transcript: text}]; value.isFinal = true; return {resultIndex: 0, results: [value]};}
const stopped = setup();
stopped.elements.mic.onclick();
assert.equal(stopped.sendButton.disabled, true);
assert.equal(stopped.elements.status.textContent, 'Listening…');
stopped.instance.onresult(result('I have a headache'));
assert.equal(stopped.transcripts().length, 0, 'Recording must not submit a chat or transcript automatically');
stopped.elements.stop.onclick();
assert.equal(stopped.transcripts()[0].value.text, 'I have a headache');
assert.equal(stopped.sendButton.disabled, false);
const cancelled = setup(); cancelled.elements.mic.onclick(); cancelled.instance.onresult(result('Discard these words')); cancelled.elements.close.onclick();
assert.equal(cancelled.transcripts().length, 0);
assert.equal(cancelled.sendButton.disabled, false);
const denied = setup(); denied.elements.mic.onclick(); denied.instance.onerror({error: 'not-allowed'});
assert.match(denied.elements.preview.textContent, /Microphone access was not allowed/);
assert.equal(denied.transcripts().length, 0); assert.equal(denied.sendButton.disabled, false);
const unsupported = setup(false); unsupported.elements.mic.onclick();
assert.match(unsupported.elements.preview.textContent, /not supported/);
assert.equal(unsupported.transcripts().length, 0);
const empty = setup(); empty.elements.mic.onclick(); empty.elements.stop.onclick();
assert.equal(empty.transcripts().length, 0); assert.match(empty.elements.preview.textContent, /No speech was detected/);
const nested = setup(false, true); nested.elements.mic.onclick();
assert.equal(nested.classes.has('voice-expanded'), true, 'Panel must expand even when frameElement is unavailable');
nested.elements.close.onclick();
assert.equal(nested.classes.has('voice-expanded'), false);
const abortError = setup(); abortError.elements.mic.onclick();
abortError.instance.abort = () => {throw new Error('Microphone has not started');};
abortError.elements.close.onclick();
assert.equal(abortError.classes.has('voice-expanded'), false);
assert.equal(abortError.sendButton.disabled, false);
console.log('Voice component: stop, cancel, permission denial, unsupported browser and empty speech passed.');
