/**
 * @file Sample CLI tool for rendering a Mii icon with body
 * headlessly using Node.js and THREE.WebGPURenderer.
 * Before running, install: npm install webgpu three
 *
 * Currently, this requires acquiring body models yourself.
 * Search "BodyPath" to see where to get them and replace their paths.
 * @author Arian Kordi <https://github.com/ariankordi>
 */
// @ts-check

// Specific to this example.
import * as fs from 'node:fs/promises';
import path from 'path';
import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';
import * as THREE from 'three/webgpu'; // THREE.*, WebGPURenderer
// Dependencies for body scaling.
import { GLTFLoader, SkeletonUtils } from 'three/examples/jsm/Addons.js';
import { addSkeletonScalingExtensions } from '../helpers/SkeletonScalingExtensions.js';
import { detectModelDesc } from '../helpers/ModelScaleDesc.js';
// FFL.js imports.
import { addWebGPUExtensions, createThreeRenderer, encodeBmpImage } from '../helpers/HeadlessWebGPU.js';
import {
	prepareBodyForCharModel,
	attachHeadToBody, disposeModel,
	adjustCameraForBodyHead,
	getFaceCamera, getWholeBodyCamera
} from '../helpers/BodyUtilities.js';
// Standard non-Node dependencies:
import { PNG } from 'pngjs';
import {
	FFL, CharModel, PantsColor,
	pantsColors, FFLExpression
} from '../ffl.js';
import FFLShaderNodeMaterial from '../materials/FFLShaderNodeMaterial.js';
import ModuleFFL from './ffl-emscripten-single-file.cjs';
/* globals process Buffer -- Node.js */
// polyfill.js
import { create, globals } from 'webgpu';

// Copy the WebGPU interfaces into global scope
Object.assign(globalThis, globals);

// Safely define or extend the navigator object
if (!globalThis.navigator) {
	Object.defineProperty(globalThis, 'navigator', {
		value: { userAgent: 'node' },
		writable: true,
		configurable: true
	});
}

Object.defineProperty(globalThis.navigator, 'gpu', {
	value: create([]),
	writable: true,
	configurable: true
});

// Imports for standalone JSDoc types.
/**
 * @typedef {import('../materials/SampleShaderMaterial.js').SampleShaderMaterialColorInfo} SampleShaderMaterialColorInfo
 * @typedef {import('../helpers/SkeletonScalingExtensions.js').SkeletonWithAttachments} SkeletonWithAttachments
 * @typedef {import('../helpers/ModelScaleDesc.js').ModelScaleDesc} ModelScaleDesc
 * @typedef {import('../helpers/BodyUtilities.js').BodyModel} BodyModel
 * @typedef {import('../ffl.js').MaterialConstructor} MaterialConstructor
 */

/**
 * Parses either hex or Base64 -> U8.
 * Additionally strips spaces from the input.
 * @param {string} text - Input encoded text.
 * @returns {Uint8Array} Decoded bytes.
 */
function parseHexOrBase64ToBytes(text) {
	text = text.replace(/\s+/g, ''); // Strip spaces.
	// Check if it is hex, otherwise assume it is Base64.
	return /^[0-9a-fA-F]+$/.test(text)
		? Buffer.from(text, 'hex')
		: Buffer.from(text, 'base64');
}

// // ---------------------------------------------------------------------
// //  Body Constants
// // ---------------------------------------------------------------------

// URLs for body models.
// TODO: We should have a script to generate these from the original files.
// Until then, these need to be acquired and placed manually.
// Models from Mii Creator may work, or from here: https://github.com/ariankordi/ffl-raylib-samples/tree/master/models
const maleBodyPath = path.resolve('./miibodymiddle male test.glb');
const femaleBodyPath = path.resolve('./miibodymiddle female test.glb');

const pantsColorDefault = pantsColors[PantsColor.RedRegular].clone().convertLinearToSRGB();

/** @typedef {import('three/examples/jsm/loaders/GLTFLoader.js').GLTF} GLTF */
/**
 * Contains body models for each gender. The glTFs contain animations.
 * @type {Record<import('../ffl.js').FFLGender, GLTF>}
 */
const bodyTemplates = [];

// // ---------------------------------------------------------------------
// //  Model Loading Helpers
// // ---------------------------------------------------------------------

/**
 * Async wrapper to load a GLTF model from a file path.
 * @param {string} path - The file path to load the glTF model from.
 * @returns {Promise<GLTF>}
 * The GLTF object. `gltf.scene` contains the mesh group.
 */
async function loadGLTFFromFS(path) {
	const content = await fs.readFile(path);
	return new Promise((resolve, reject) => {
		new GLTFLoader().parse(/** @type {ArrayBuffer} */ (content.buffer),
			', ', resolve, reject);
	});
}

/**
 * @param {Record<number, GLTF>} templates - Array of models to fill.
 * @returns {Promise<void>} Completes when the models are loaded into the array.
 */
const preloadBodyTemplates = templates =>
	Promise.all([loadGLTFFromFS(maleBodyPath),
		loadGLTFFromFS(femaleBodyPath)]).then((values) => {
		templates[0] = values[0];
		templates[1] = values[1];
	});

/**
 * Loads the body model for the specified gender from the glTF templates.
 * Applies a single animation from the glTF.
 * @param {number} gender - The gender for the model.
 * @returns {BodyModel} The body model.
 * @throws {Error} Throws if there is no model for the `genderVal`.
 * @todo This should probably be reduced to not load the animation,
 * so that it's in the caller's control if we want to swap animations.
 * It would have to return the GLTF though so the caller uses .scene/.animations.
 */
function loadBodyModel(gender) {
	const gltf = bodyTemplates[gender];
	if (!gltf) {
		throw new Error(`No body model loaded for gender ${gender}`);
	}

	// Use this method to clone model with proper skinning.
	const model = SkeletonUtils.clone(gltf.scene);

	// Assign one animation from the glTF model.
	const animations = gltf.animations;
	const mixer = new THREE.AnimationMixer(model);
	if (animations.length) {
		// Use the animation named 'Wait' if it exists, otherwise choose the first one.
		let clip = animations.find(a => a.name === 'Wait');
		if (!clip) {
			clip = animations[0];
		}

		mixer.clipAction(clip).play()
			.setLoop(THREE.LoopRepeat, Infinity);
	}

	return {
		model, animations, mixer,
		scaleDesc: detectModelDesc(model)
	};
}

// // ---------------------------------------------------------------------
// //  Rendering Function
// // ---------------------------------------------------------------------

/**
 * @typedef {Object} IconRenderRequest
 * @property {Uint8Array} data
 * @property {number} width
 * @property {boolean} isAllBody
 * @property {number} expression
 */

/**
 * @param {THREE.Renderer} renderer - The Three.js renderer.
 * @param {FFL} ffl - The FFL.js instance.
 * @param {MaterialConstructor} matClass - The material class for the head and body.
 * @param {IconRenderRequest} request - Parameters for rendering the icon.
 * @returns {Promise<Uint8Array>} Rendered image as image file data.
 */
async function renderRequestToImage(renderer, ffl, matClass, request, format, size, name, expression, mii_data) {
	const scene = new THREE.Scene();
	// Scene should have a transparent background.
	// Usually, it's transparent white (FFFFFF00) - then colored glasses
	// are fully accurate. I tried it and it didn't work though, maybe glass blending options are off?

	/** @type {import('../ffl.js').CharModel|null} */ let charModel = null;
	/** @type {BodyModel|null} */ let body = null;

	try {
		// Create Mii model and add to the scene.
		charModel = new CharModel(ffl, request.data,
			request.expression, matClass, renderer);

		// TODO: In certain instances (charline), nose and forehead
		// may disappear due to frustum culling. This is probably
		// from mask/glass not being properly marked transparent?
		// For now, disable frustumCulled on every head mesh.
		charModel.meshes.traverse((m) => {
			m.frustumCulled = false;
		});

		body = loadBodyModel(charModel.charInfo.gender);
		const bodyMdl = body.model;

		body.mixer.update(0); // Set animation to 0th frame.
		// Set uniforms on the new body model, and attach the head to it.
		prepareBodyForCharModel(body, matClass,
			charModel.favoriteColor,
			charModel.getBodyScale(),
			pantsColorDefault);

		attachHeadToBody(body, charModel.meshes);
		scene.add(bodyMdl); // Also adds the head model to the scene.

		// Use the camera for an icon pose.
		/** @type {THREE.PerspectiveCamera} */ let camera;
		if (request.isAllBody) { // whole_body/all_body camera.
			// TODO: Should eventually use 270x360 (3:4) aspect ratio like real account renders.
			camera = getWholeBodyCamera(1, charModel.charInfo.height);
		} else { // Face only render.
			camera = getFaceCamera();
			adjustCameraForBodyHead(camera, body);
		}

		// Render the scene, and read the pixels into a buffer.
		const rt = new THREE.RenderTarget(request.width, request.width, {
			samples: 4, // Uncomment for antialiasing.
			minFilter: THREE.LinearFilter,
			magFilter: THREE.LinearFilter
		});
		renderer.setRenderTarget(rt); // Set new target.
		// rt.viewport = new THREE.Vector4(0, 0, 200, 200);
		renderer.setClearColor(0x000000, 0); 
		renderer.render(scene, camera);
		const pixels = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, rt.width, rt.height);
		// NObody likes bmp
let imageData; 
const srcWidth = Math.ceil(rt.width / 64) * 64;
const srcHeight = Math.ceil(rt.height / 64) * 64;
const storageDir = path.resolve(import.meta.dirname, '../src/storage/miis');
const filePath = path.join(storageDir, `${name}-${mii_data}`, String(expression), `icon-${size}.${format}`);
const folderPath = path.dirname(filePath);
await mkdir(folderPath, { recursive: true });

if (format === 'png') {
    imageData = await sharp(Buffer.from(pixels), {
        raw: {
            width: srcWidth,
            height: srcHeight,
            channels: 4
        }
    })
    .resize(size, size, { fit: 'fill' })
    .png()
    .toBuffer();

    await fs.writeFile(filePath, imageData);
} else if (format === 'jpeg' || format === 'jpg') {
    imageData = await sharp(Buffer.from(pixels), {
        raw: {
            width: srcWidth,
            height: srcHeight,
            channels: 4
        }
    })
    .resize(size, size, { fit: 'fill' })
    .flatten({ background: { r: 255, g: 255, b: 255 } })
    .jpeg({ quality: 90, mozjpeg: true })
    .toBuffer();

    await fs.writeFile(filePath, imageData);
} else if (format === 'tga') {
    const resizedRawPixels = await sharp(Buffer.from(pixels), {
        raw: {
            width: srcWidth,
            height: srcHeight,
            channels: 4
        }
    })
    .resize(size, size, { fit: 'fill' })
    .raw()
    .toBuffer();

    const mutablePixels = Buffer.from(resizedRawPixels);
    const tgaHeader = Buffer.alloc(18);
    tgaHeader.writeUInt8(0, 0);       
    tgaHeader.writeUInt8(0, 1);       
    tgaHeader.writeUInt8(2, 2); 
    tgaHeader.writeUInt16LE(size, 12);
    tgaHeader.writeUInt16LE(size, 14);
    tgaHeader.writeUInt8(32, 16);
    tgaHeader.writeUInt8(40, 17); 

    for (let i = 0; i < mutablePixels.length; i += 4) {
        const r = mutablePixels[i];
        const b = mutablePixels[i + 2];
        mutablePixels[i] = b;     
        mutablePixels[i + 2] = r; 
    }

    imageData = Buffer.concat([tgaHeader, mutablePixels]);
    await fs.writeFile(filePath, imageData);
} else {
	    const png = new PNG({ width: rt.width, height: rt.height });
    png.data = Buffer.from(pixels.buffer);
    imageData = PNG.sync.write(png); 
}

return imageData;

	} finally {
		charModel && charModel.dispose(); // Mii model
		if (body) {
			disposeModel(body.model); // Body model
			body.mixer.uncacheRoot(body.model);
		}
	}
}

// // ---------------------------------------------------------------------
// //  CLI Entrypoint
// // ---------------------------------------------------------------------

/** Entrypoint */
async function main( mii_data, type, expression, format, size, name) {




	const width = Number.parseInt('512', 10);
	// const height = width;
	const isAllBody = type === 'all_body';

    

	// Validate inputs. Mii data is validated by rendering functions.
	if (Number.isNaN(width) || width <= 0) {
		console.error('Width must be a positive integer.');
		return false
	}

	if (width % 64 !== 0) {
		console.error('The resolution needs to be a multiple of 64. If it\'s not, the pixels will have this weird padding applied that doesn\'t play nice with image encoders. Examples: 128, 192, 256');
		return false
	}

	if (Number.isNaN(expression) || expression < 0 || expression >= FFLExpression.MAX) {
		console.error(`Expression must be in range: 0-${FFLExpression.MAX - 1}.`);
		return false
	}
	const resourceFile = await fs.readFile('./FFLResHigh.dat');

	addSkeletonScalingExtensions(THREE.Skeleton);
	addWebGPUExtensions(); // Add WebGPU extensions needed for Node.js.

	/** The renderer. Dimensions are not passed, because render targets are used. */
	const renderer = await createThreeRenderer();
	/** Reassign the device lost handler to do nothing. */
	renderer.onDeviceLost = () => { };

	try {
		await preloadBodyTemplates(bodyTemplates); // Load body models.
	} catch (e) {
		console.error('Could not load body models. TODO, as of right now this script needs you to acquire them manually. Search for "BodyPath = ..." in the script for details. Sorry.');
		throw e;
	}

	/** @type {FFL|null} */ let ffl = null;
	// TODO: Why doesn't this type show up properly?
	const matClass = /** @type {MaterialConstructor} */
		(/** @type {unknown} */ (FFLShaderNodeMaterial));

	try {
		// Initialize FFL, which loads resource into memory.
		ffl = await FFL.initWithResource(resourceFile, ModuleFFL);
		ffl.setRenderer(renderer); // Tell FFL.js we are WebGPU

		// Go through all Mii data inputs.


			console.info(`Rendering Mii`);
			/** @type {IconRenderRequest} */ const request = {
				data: parseHexOrBase64ToBytes(mii_data),
				width, isAllBody, expression
			};
			let imageData = await renderRequestToImage(renderer, ffl, matClass, request, format, size, name, expression, mii_data);

			return imageData
		
	} finally {
		// Clean up.
		(ffl) && ffl.dispose(); // Free FFL resource from WASM heap.
		renderer.dispose(); // Dispose Three.js renderer.

		// Destroy the GPUDevice.
		const device = /** @type {Object<string, *>} */ (renderer.backend).device;
		if (device instanceof GPUDevice) {
			await device.queue.onSubmittedWorkDone();
			device.destroy();
		}
	}
}

// eslint-disable-next-line unicorn/prefer-top-level-await -- needs lib = es2017
export {main}
export default {main}
