import { mount } from 'svelte';
import App from './App.svelte';
import { clearIsolationGuard, ensureIsolation } from './coi';
import './ui/app.css';

clearIsolationGuard();
ensureIsolation();
mount(App, { target: document.getElementById('app')! });
