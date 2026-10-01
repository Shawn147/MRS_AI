// Simulate browser file drops; no personal files or browser permissions are used.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const script = fs.readFileSync('assets/file_drop/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function setup({remaining=3, disabled=false}={}) {
 const messages=[], listeners={}, windowListeners={}, classes=new Set(), button={disabled:false};
 const composer={classList:{toggle(name,on){on?classes.add(name):classes.delete(name);}}};
 const document={querySelector(selector){return selector.includes('button')?button:composer;},
  addEventListener(name,fn){listeners[name]=fn;}, removeEventListener(name){delete listeners[name];}};
 const parent={document,postMessage(message){messages.push(message);}};
 const window={parent,addEventListener(name,fn){windowListeners[name]=fn;}};
 class FileReader {
  readAsDataURL(file){if(file.fail){this.onerror();return;}this.result='data:text/plain;base64,'+Buffer.from(file.text||'Lab result').toString('base64');this.onload();}
 }
 vm.runInNewContext(script,{window,FileReader,Date,Math});
 windowListeners.message({source:parent,data:{type:'streamlit:render',args:{chat_id:'test-chat',remaining,disabled,max_bytes:10*1024*1024}}});
 const event=(files,inside=true,types=['Files'])=>({target:{closest(){return inside?composer:null;}},dataTransfer:{files,types},preventDefault(){this.prevented=true;},stopPropagation(){this.stopped=true;}});
 return {classes,button,listeners,event,windowListeners,values:()=>messages.filter(m=>m.type==='streamlit:setComponentValue').map(m=>m.value)};
}
(async()=>{
 const test=setup(), file={name:'sample.txt',size:20,text:'Hemoglobin 10.2 g/dL'};
 const over=test.event([file]);test.listeners.dragover(over);assert(test.classes.has('file-drag-over'));assert.equal(over.dataTransfer.dropEffect,'copy');
 await test.listeners.drop(test.event([file]));
 assert.equal(test.values()[0].type,'files');assert.equal(Buffer.from(test.values()[0].files[0].content,'base64').toString(),file.text);
 assert.equal(test.values()[0].chat_id,'test-chat');assert.equal(test.button.disabled,false);assert(!test.classes.has('file-drag-over'));
 for(const [files,remaining,match] of [[[file],0,/3 files/],[[{name:'bad.exe',size:1}],3,/Supported files/],[[{name:'large.pdf',size:11*1024*1024}],3,/10 MB/],[[{name:'empty.txt',size:0}],3,/nonempty/],[[{name:'failed.pdf',size:10,fail:true}],3,/could not be read/]]){
  const bad=setup({remaining});await bad.listeners.drop(bad.event(files));assert.equal(bad.values()[0].type,'error');assert.match(bad.values()[0].message,match);assert.equal(bad.button.disabled,false);
 }
 const disabled=setup({disabled:true});await disabled.listeners.drop(disabled.event([file]));assert.equal(disabled.values().length,0);
 const outside=setup();await outside.listeners.drop(outside.event([file],false));assert.equal(outside.values().length,0);
 const text=setup(), textDrop=text.event([],true,['text/plain']);await text.listeners.drop(textDrop);assert(!textDrop.prevented);
 test.windowListeners.beforeunload();assert(!test.listeners.drop);assert(!test.listeners.dragend);
 console.log('File drop component: valid drop, highlight, limits, read failure, disabled/outside/text drops and cleanup passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
