#!/usr/bin/env python3
"""
generate_boards.py — generate synthetic chess board PNGs (stdlib only, no Pillow)
Usage: python3 scripts/generate_boards.py --count 200 --output data/boards_200
"""
import struct, zlib, os, random, argparse

def write_png(filename, pixels, w, h):
    def chunk(name, data):
        c = name+data
        return struct.pack('>I',len(data))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
    raw = b''.join(b'\x00'+bytes(r) for r in pixels)
    with open(filename,'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                +chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
                +chunk(b'IDAT',zlib.compress(raw,6))
                +chunk(b'IEND',b''))

LIGHT=(240,217,181); DARK=(181,136,99); BP=(35,28,20); WP=(248,243,230)

def gen(seed, size=512):
    rng = random.Random(seed); sq=size//8
    pix = []
    for py in range(size):
        row=[]
        for px in range(size):
            row.extend(LIGHT if ((py//sq)+(px//sq))%2==0 else DARK)
        pix.append(row)
    for _ in range(rng.randint(8,24)):
        pr,pf=rng.randint(0,7),rng.randint(0,7)
        cx,cy=pf*sq+sq//2,pr*sq+sq//2; r=sq//3; r2=r*r
        col=list(BP if rng.random()<0.5 else WP)
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                if dx*dx+dy*dy<=r2:
                    x,y=cx+dx,cy+dy
                    if 0<=x<size and 0<=y<size:
                        pix[y][x*3:x*3+3]=col
    return pix

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--size', type=int,default=512)
    p.add_argument('--output',default='data/sample_boards')
    a=p.parse_args()
    os.makedirs(a.output,exist_ok=True)
    print(f'Generating {a.count} boards ({a.size}x{a.size}) → {a.output}/')
    for i in range(a.count):
        write_png(f'{a.output}/board_{i:03d}.png', gen(i*31+7,a.size), a.size, a.size)
        if (i+1)%10==0 or i==a.count-1: print(f'  {i+1}/{a.count}')
    print('Done.')

if __name__=='__main__': main()
