// Good Cream Color #F6FEDC

import React, { useRef, useMemo, useState, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { Mesh, ShaderMaterial, Matrix4 } from "three";
import vertShader from "../../shaders/vertex.glsl";
import fragShader from "../../shaders/fragment.glsl";
import sphereIntersectionShader from '../../shaders/sifragment.glsl';

import { BoxBufferGeometry } from "three";
import { useControls } from "leva";

import * as THREE from "three";

import mcap from "../../../static/7.png";

interface SceneProps {
    color: string;
    hoverColor: string;
}

interface BoxProps {
    color: string;
    hoverColor: string;
}

const Box = (props: JSX.IntrinsicElements["mesh"]) => {

    const clamp = (a: number, min = 0, max = 1) =>
        Math.min(max, Math.max(min, a));
    const invlerp = (x: number, y: number, a: number) =>
        clamp((a - x) / (y - x));

    // Refs
    const shaderMatRef = useRef<ShaderMaterial>();
    const cubeRef = useRef<THREE.Mesh>(null!);

    // State
    const mouse = useThree((state) => state.mouse);

    // Controls
    const {
        fovNum,
        scaleX,
        scaleY,
        scaleZ,
        steps,
        iterations,
        power,
        bailout,
        lightIntensity,
        sinScale,
        glowBrightness,
        glowThickness,
        color1,
        color2,
    } = useControls({
        fovNum: {
            value: 75,
            min: 75,
            max: 110,
            step: 1,
        },
        scaleX: {
            value: 2,
            min: 0,
            max: 8,
            step: 1.0,
        },
        scaleY: {
            value: 2,
            min: 0,
            max: 8,
            step: 1.0,
        },
        scaleZ: {
            value: 2,
            min: 0,
            max: 8,
            step: 1.0,
        },
        steps: {
            value: 450,
            min: 25,
            max: 1000,
            step: 1.0,
        },
        iterations: {
            value: 15,
            min: 0,
            max: 20,
            step: 0.001,
        },
        power: {
            value: 10,
            min: -50,
            max: 50,
            step: 0.001,
        },
        bailout: {
            value: 2,
            min: -8,
            max: 8,
            step: 0.001,
        },
        lightIntensity: {
            value: 0.45,
            min: 0.1,
            max: 1.0,
            step: 0.001,
        },
        sinScale: {
            value: 5.0,
            min: 1.0,
            max: 20,
            step: 0.001,
        },
        glowBrightness: {
            value: 0.5,
            min: 0.0,
            max: 10,
            step: 0.001,
        },
        glowThickness: {
            value: 0.5,
            min: 0.0,
            max: 50,
            step: 0.001,
        },
        color1: { a: 1.0, g: 28.0, b: 214.0, r: 0.0 },
        color2: { a: 1.0, g: 206.0, b: 125.0, r: 197.0 },
    });

     const {gl} =  useThree();

     console.log(gl);

    // need fov and aspect ratio for shader
    

    // Textures
    const matcap = useMemo(() => new THREE.TextureLoader().load(mcap), [mcap]);

    const uniforms = useMemo(() => {
        return {
            uTime: { value: 0 },
            matcap: { value: matcap },
            mouse: { value: mouse },
            uSteps: { value: 450 },
            uIterations: { value: 15 },
            uPower: { value: 10 },
            uBailout: { value: 2 },
            uLightIntensity: { value: 0.45 },
            uSinScale: { value: 5.0 },
            uGlowBrightness: { value: 0.5 },
            uGlowThickness: { value: 0.5 },

            uColor1R: { value: 0.0 },
            uColor1G: { value: 0.0 },
            uColor1B: { value: 0.0 },

            uColor2R: { value: 0.0 },
            uColor2G: { value: 0.0 },
            uColor2B: { value: 0.0 },
        };
    }, [mouse]);

    useFrame((_) => {
        let time = _.clock.getElapsedTime();
        uniforms.uTime.value = time;
    });

    // Attributes
    const geometry = useMemo(
        () => new BoxBufferGeometry(scaleX, scaleY, scaleZ),
        [scaleX, scaleY, scaleZ]
    );

    if (shaderMatRef.current) {
        shaderMatRef.current.uniforms.uSteps.value = steps;
        shaderMatRef.current.uniforms.uIterations.value = iterations;
        shaderMatRef.current.uniforms.uPower.value = power;
        shaderMatRef.current.uniforms.uBailout.value = bailout;
        shaderMatRef.current.uniforms.uLightIntensity.value = lightIntensity;
        shaderMatRef.current.uniforms.uSinScale.value = sinScale;
        shaderMatRef.current.uniforms.uGlowBrightness.value = glowBrightness;
        shaderMatRef.current.uniforms.uGlowThickness.value = glowThickness;

        shaderMatRef.current.uniforms.uColor1R.value = invlerp(0.0, 255.0, color1.r);
        shaderMatRef.current.uniforms.uColor1R.value = invlerp(0.0, 255.0, color1.g);
        shaderMatRef.current.uniforms.uColor1R.value = invlerp(0.0, 255.0, color1.b);
        
        shaderMatRef.current.uniforms.uColor2R.value = invlerp(0.0, 255.0, color2.r);
        shaderMatRef.current.uniforms.uColor2R.value = invlerp(0.0, 255.0, color2.g);
        shaderMatRef.current.uniforms.uColor2R.value = invlerp(0.0, 255.0, color2.b);
    }

    return (
        <mesh
            ref={cubeRef}
            scale={[1, 1, 1]}
            geometry={geometry}
            position={[0.0, 0.0, 0.0]}
        >
            <shaderMaterial
                ref={shaderMatRef}
                vertexShader={vertShader}
                fragmentShader={sphereIntersectionShader}
                uniforms={uniforms}
                side={THREE.BackSide}
            />
        </mesh>
    );
};

export default function Scene(props: SceneProps) {
    return (
        <>
            <ambientLight />
            <pointLight position={[10, 10, 10]} />
            <Box />
            <OrbitControls />
        </>
    );
}
