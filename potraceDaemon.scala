import java.nio.file._, StandardWatchEventKinds._
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters._
import scala.sys.process._
import java.io.FileOutputStream
import java.nio.channels.FileChannel
import java.util.concurrent.{Executors, TimeUnit, TimeoutException, ExecutorService}

@main def potraceDaemon(inDir: String, outDir: String): Unit =
  val (in, out) = (Paths.get(inDir), Paths.get(outDir))
  val watcher = in.getFileSystem.newWatchService()
  in.register(watcher, ENTRY_CREATE)
  Files.createDirectories(out)
  
  println(s"[*] Edge detection daemon started. Watching: $in -> Output: $out")
  
  var imageExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  
  def isFileReady(path: Path): Boolean =
    try
      val channel = FileChannel.open(path, StandardOpenOption.WRITE)
      val lock = channel.tryLock()
      if lock != null then
        lock.release()
        channel.close()
        true
      else
        channel.close()
        false
    catch case _: Exception => false

  def processFile(file: Path): Unit =
    if Files.isRegularFile(file) && !file.getFileName.toString.startsWith(".") then
      println(s"[*] Detected file: ${file.getFileName}. Verifying lock availability...")
      
      var checks = 0
      while !isFileReady(file) && checks < 20 do
        Thread.sleep(250)
        checks += 1

      if !isFileReady(file) then
        println(s"[!] Warning: File ${file.getFileName} is locked or incomplete after timeout. Skipping.")
        return

      val decodeTask = imageExecutor.submit(() => Option(ImageIO.read(file.toFile)))

      try
        val imgOpt = decodeTask.get(5, TimeUnit.SECONDS)
        imgOpt match
          case Some(img) =>
            val (w, h) = (img.getWidth, img.getHeight)
            val pixels = img.getRGB(0, 0, w, h, null, 0, w)
            
            val pbm = Files.createTempFile("edge", ".pbm")
            val outStream = new FileOutputStream(pbm.toFile)
            try
              outStream.write(s"P4\n$w $h\n".getBytes("US-ASCII"))
              for y <- 0 until h do
                var byteVal = 0
                var bitCount = 0
                val rowOffset = y * w
                for x <- 0 until w do
                  val gx = if x > 0 && x < w - 1 then (pixels(rowOffset + x + 1) & 0xff) - (pixels(rowOffset + x - 1) & 0xff) else 0
                  val gy = if y > 0 && y < h - 1 then (pixels((y + 1) * w + x) & 0xff) - (pixels((y - 1) * w + x) & 0xff) else 0
                  val bit = if math.hypot(gx, gy) > 60 then 1 else 0
                  
                  byteVal = (byteVal << 1) | bit
                  bitCount += 1
                  if bitCount == 8 then
                    outStream.write(byteVal & 0xFF)
                    byteVal = 0
                    bitCount = 0
                    
                if bitCount > 0 then
                  outStream.write((byteVal << (8 - bitCount)) & 0xFF)
            finally outStream.close()
            
            val svg = Files.createTempFile("edge", ".svg")
            
            // Safe process invocation (prevents shell injection)
            val cmd = Seq("potrace", "-s", "-o", svg.toAbsolutePath.toString, pbm.toAbsolutePath.toString)
            val exitCode = cmd.!
            
            if exitCode == 0 then
              val outFile = out.resolve(s"${file.getFileName}.svg")
              Files.copy(svg, outFile, StandardCopyOption.REPLACE_EXISTING)
              println(s"[+] Successfully generated SVG: ${file.getFileName}.svg")
            else
              println(s"[!] Potrace failed with exit code $exitCode for ${file.getFileName}")

            Files.deleteIfExists(pbm)
            Files.deleteIfExists(svg)
            
          case None =>
            println(s"[!] Warning: ImageIO returned null for ${file.getFileName}. Skipping.")
      catch
        case _: TimeoutException =>
          decodeTask.cancel(true)
          imageExecutor.shutdownNow()
          imageExecutor = Executors.newSingleThreadExecutor() // Re-initialize thread pool on un-interruptible lock
          println(s"[!] Error: Image decoding timed out for ${file.getFileName}. Resetting worker thread.")
        case e: Exception =>
          println(s"[!] Error processing ${file.getFileName}: ${e.getMessage}")

  // Scan pre-existing files on startup
  try
    val stream = Files.newDirectoryStream(in)
    try for f <- stream.asScala do processFile(f)
    finally stream.close()
  catch case e: Exception => println(s"[!] Startup directory scan failed: ${e.getMessage}")

  // Watch key event loop
  while true do
    val key = watcher.take()
    try
      for e <- key.pollEvents().asScala if e.kind() == ENTRY_CREATE do
        val file = in.resolve(e.context.asInstanceOf[Path])
        processFile(file)
    finally
      key.reset()